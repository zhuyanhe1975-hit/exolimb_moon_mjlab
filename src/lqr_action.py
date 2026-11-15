"""60Hz residual position action with a fixed local LQR, batched on device.

MuJoCo Warp's native quat_sub supplies the same tangent convention as
mj_differentiatePos. No CPU contact loops, inverse dynamics or Riccati solve
occurs in the training loop. The native free-root angular qvel is BODY-frame.
"""
from dataclasses import dataclass
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.math import mul_quat, quat_sub, rot_vec_quat
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

from lqr_artifact import DEFAULT_FILE, KEEP, NV, NU, STATE_DIM, load_artifact
from lunar_terrain import surface


@wp.kernel
def tangent_state(q: wp.array2d(dtype=float), v: wp.array2d(dtype=float),
                  heading: wp.array2d(dtype=float), height: wp.array(dtype=float),
                  ref: wp.array(dtype=float), out: wp.array2d(dtype=float)):
    i = wp.tid()
    # Remove only the randomized world yaw. Do not reconstruct dynamics or COM.
    h_inv = wp.quat(heading[i,0], -heading[i,1], -heading[i,2], -heading[i,3])
    local_q = mul_quat(h_inv, wp.quat(q[i,3],q[i,4],q[i,5],q[i,6]))
    ref_q = wp.quat(ref[3],ref[4],ref[5],ref[6])
    # Spatial tangent in the yaw-local WORLD frame. If the equilibrium leans,
    # dropping a BODY-z rotation would not remove the world-yaw gauge.
    difference = rot_vec_quat(quat_sub(local_q,ref_q),ref_q)
    out[i,0] = 0.  # horizontal translation is a discarded gauge, not an objective
    out[i,1] = 0.
    out[i,2] = q[i,2]-height[i]-ref[2]
    for j in range(3):
        out[i,3+j] = difference[j]
    for j in range(23):
        out[i,6+j] = q[i,7+j]-ref[7+j]
    linear = rot_vec_quat(wp.vec3(v[i,0],v[i,1],v[i,2]), h_inv)
    for j in range(3):
        out[i,29+j] = linear[j]
    angular = rot_vec_quat(wp.vec3(v[i,3],v[i,4],v[i,5]),local_q)
    for j in range(3):
        out[i,32+j] = angular[j]
    for j in range(6,29):
        out[i,29+j] = v[i,j]


def apply_residual(base, residual, lower, upper, residual_scale):
    requested = base+residual.clamp(-2.,2.)*residual_scale
    final = requested.clamp(lower,upper)
    saturation = (requested-final).abs().gt(1e-6).any(-1).float()
    return final, saturation


def blend_commands(dx, K, ctrl0, default, load_ratio, slope, residual,
                   lower, upper, residual_scale, fallback_scale=.25):
    """Pure tensor math, also used by CPU unit checks (no environment/physics)."""
    # Full authority only with meaningful support and near the flat reference.
    # This gain was derived for DOUBLE support. One loaded foot is not enough.
    contact = ((load_ratio.amin(-1)-.02)/.08).clamp(0.,1.)
    contact = contact*(load_ratio.sum(-1)/.5).clamp(0.,1.)
    tilt = ((.35-dx[:,1:3].norm(dim=-1))/.20).clamp(0.,1.)
    joints = ((.60-dx[:,3:26].abs().amax(-1))/.25).clamp(0.,1.)
    slope_gate = ((np.deg2rad(12.)-slope)/np.deg2rad(9.)).clamp(0.,1.)
    alpha = contact*tilt*joints*slope_gate
    raw = ctrl0-dx@K.T
    requested_base = default+alpha[:,None]*(raw-default)
    base = requested_base.clamp(lower,upper)
    scale=(residual_scale+(1.-alpha)*(fallback_scale-residual_scale))[:,None]
    final, saturation = apply_residual(base,residual,lower,upper,scale)
    saturation = torch.maximum(saturation,(requested_base-base).abs().gt(1e-6).any(-1).float())
    return base, final, alpha, saturation


@dataclass(kw_only=True)
class ResidualLqrActionCfg(JointPositionActionCfg):
    # Resolve the artifact at ENV construction, not task registration. This lets
    # playback choose the controller snapshot adjacent to its checkpoint.
    residual_scale: float = .05
    fallback_scale: float = .25

    def build(self, env):
        return ResidualLqrAction(self, env)


class ResidualLqrAction(JointPositionAction):
    def __init__(self, cfg, env):
        if cfg.scale != 1. or cfg.offset != 0. or not cfg.use_default_offset:
            raise ValueError('Residual LQR uses residual_scale/fallback_scale and the native default pose; '
                             'generic scale/offset overrides are unsupported')
        if not 0. < cfg.residual_scale <= cfg.fallback_scale <= .25:
            raise ValueError('Expected 0 < residual_scale <= fallback_scale <= 0.25 rad/unit')
        super().__init__(cfg, env)
        self.artifact_path = Path(os.environ.get('WANGYU_LQR_FILE',str(DEFAULT_FILE))).expanduser().resolve()
        artifact, self.metadata, self.sha256 = load_artifact(self.artifact_path)
        names = tuple(self._entity.joint_names)
        if names != tuple(artifact['joint_names']) or tuple(self.target_names) != names:
            raise ValueError('LQR native joint/state ordering differs from the entity action ordering')
        if (env.cfg.decimation != 2 or not np.isclose(env.physics_dt,1/120)
                or not np.isclose(env.cfg.sim.mujoco.gravity[2],-9.81/6)):
            raise ValueError('LQR was prepared for 120Hz physics, 60Hz control and lunar gravity')
        model = env.sim.mj_model
        actuator_rows = {name:i for i,name in enumerate(artifact['actuator_joints'])}
        rows = np.array([actuator_rows[name] for name in names])
        actual_ids = [model.actuator('robot/'+name).id for name in names]
        if any(int(model.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT)
               for i in actual_ids):
            raise ValueError('LQR requires direct native joint transmissions')
        for actual, expected in ((model.actuator_gainprm[actual_ids,0],artifact['kp'][rows]),
                                 (-model.actuator_biasprm[actual_ids,1],artifact['kp'][rows]),
                                 (-model.actuator_biasprm[actual_ids,2],artifact['kd'][rows]),
                                 (-model.actuator_forcerange[actual_ids,0],artifact['force_limit'][rows]),
                                 (model.actuator_forcerange[actual_ids,1],artifact['force_limit'][rows])):
            np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(model.actuator_gear[actual_ids,0],1.)
        np.testing.assert_allclose(model.actuator_gear[actual_ids,1:],0.)
        for name,value in self.metadata.get('model_options',{}).items():
            if not np.isclose(float(getattr(model.opt,name)),value,rtol=1e-7,atol=1e-12):
                raise ValueError(f'LQR physics option differs: {name}')
        joint_ids = [model.joint('robot/'+name).id for name in names]
        np.testing.assert_allclose(model.jnt_range[joint_ids],artifact['joint_range'],atol=1e-7)
        body_ids = [i for i in range(model.nbody) if model.body(i).name.startswith('robot/')]
        np.testing.assert_allclose(model.body_mass[body_ids],artifact['body_mass'][1:],atol=1e-6)
        np.testing.assert_allclose(model.opt.gravity,[0,0,-9.81/6],atol=1e-9)
        self.K = torch.tensor(artifact['K'][rows],device=self.device,dtype=torch.float32)
        self.ctrl0 = torch.tensor(artifact['ctrl0'][rows],device=self.device,dtype=torch.float32)
        if np.any(artifact['ctrl0'][rows] < np.maximum(artifact['joint_range'][:,0],-.5)) or np.any(
                artifact['ctrl0'][rows] > np.minimum(artifact['joint_range'][:,1],.5)):
            raise ValueError('LQR reference control exceeds the action target bounds')
        self.qref = torch.tensor(artifact['qpos0'],device=self.device,dtype=torch.float32)
        self.keep = torch.tensor(KEEP,device=self.device,dtype=torch.long)
        idx = self._entity.indexing
        self.qadr = torch.cat((idx.free_joint_q_adr,idx.joint_q_adr)).long()
        self.vadr = torch.cat((idx.free_joint_v_adr,idx.joint_v_adr)).long()
        self.state = torch.zeros((self.num_envs,2*NV),device=self.device)
        self.baseline = torch.zeros_like(self._raw_actions)
        self.last_command = self._offset.clone()
        self.previous_command = self._offset.clone()
        self.applied_alpha = torch.zeros(self.num_envs,device=self.device)
        self.applied_saturation = torch.zeros(self.num_envs,device=self.device)
        self.load_adr = [int(model.sensor('robot/balance_'+side+'_load').adr[0]) for side in ('left','right')]
        root = model.body('robot/Pelvis').id
        self.weight = float(model.body_subtreemass[root]*9.81/6)
        self.sample_offsets = torch.tensor([[0.,0.],[.3,0.],[-.3,0.],[0.,.3],[0.,-.3]],device=self.device)
        self.preview_alpha = self.applied_alpha.clone()
        self._preview_key = None
        print(f'[LQR] {self.artifact_path} | sha256={self.sha256} | residual={cfg.residual_scale:g}..{cfg.fallback_scale:g}rad/unit; '
              'flat local gain, contact/tilt/joint/slope gating; 60Hz feedback',flush=True)

    def preview(self, residual=None):
        # Read raw native qpos/qvel, avoiding frame fields stale by one substep.
        q = self._env.sim.data.qpos[:,self.qadr].contiguous()
        v = self._env.sim.data.qvel[:,self.vadr].contiguous()
        heading = getattr(self._env,'_stand_target_heading',self._entity.data.default_root_state[:,3:7]).contiguous()
        c = heading[:,0].square()-heading[:,3].square()
        s = 2*heading[:,0]*heading[:,3]
        offset = self.sample_offsets
        xy = torch.stack((q[:,None,0]+c[:,None]*offset[None,:,0]-s[:,None]*offset[None,:,1],
                          q[:,None,1]+s[:,None]*offset[None,:,0]+c[:,None]*offset[None,:,1]),-1)
        heights = surface(self._env).height(xy)
        ground_height = heights[:,0].contiguous()
        slope = torch.atan(torch.stack(((heights[:,1]-heights[:,2])/.6,
                                        (heights[:,3]-heights[:,4])/.6),-1).norm(dim=-1))
        wp.launch(tangent_state,dim=self.num_envs,inputs=[wp.from_torch(q),wp.from_torch(v),
                  wp.from_torch(heading),wp.from_torch(ground_height),wp.from_torch(self.qref)],
                  outputs=[wp.from_torch(self.state)],device=wp.device_from_torch(q.device))
        dx = self.state[:,self.keep]
        loads = self._env.sim.data.sensordata[:,self.load_adr].clamp_min(0.)/self.weight
        if residual is None:
            residual = torch.zeros_like(self._raw_actions)
        base, final, alpha, saturation = blend_commands(dx,self.K,self.ctrl0,self._offset,
            loads,slope,residual,self._clip[:,:,0],self._clip[:,:,1],self.cfg.residual_scale,self.cfg.fallback_scale)
        self.baseline.copy_(base)
        self.preview_alpha.copy_(alpha)
        return dx, final, alpha, saturation

    def process_actions(self, actions):
        self._raw_actions.copy_(actions)
        self.previous_command.copy_(self.last_command)
        # The actor just observed this exact post-forward state. Reuse its
        # owned baseline; no second height query, quaternion kernel or GEMM.
        if self._preview_key == observation_key(self._env):
            scale=(self.cfg.residual_scale+(1.-self._preview_authority)*
                   (self.cfg.fallback_scale-self.cfg.residual_scale))[:,None]
            final,saturation=apply_residual(self._preview_base,actions,
                self._clip[:,:,0],self._clip[:,:,1],scale)
            saturation=torch.maximum(saturation,self._preview_clipped)
            alpha=self._preview_authority
        else:
            _, final, alpha, saturation = self.preview(actions)
        self.last_command.copy_(final)
        self._processed_actions.copy_(final)
        self.applied_alpha.copy_(alpha)
        self.applied_saturation.copy_(saturation)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.
        self.last_command[ids] = self._offset[ids]
        self.previous_command[ids] = self._offset[ids]
        self._processed_actions[ids] = self._offset[ids]
        self.baseline[ids] = self._offset[ids]
        self.applied_alpha[ids] = 0.
        self.applied_saturation[ids] = 0.
        self.preview_alpha[ids] = 0.
        self._preview_key = None


def term(env):
    return env.action_manager.get_term('position')


def observation_key(env):
    return (env.common_step_counter,getattr(env,'_stand_reset_generation',0),'observation')


def observation(env):
    from balance_mdp import observation as base_observation
    key=observation_key(env)
    if getattr(env,'_lqr_observation_key',None)==key:
        return env._lqr_observation_value
    controller = term(env)
    dx, _, alpha, clipped = controller.preview()
    controller._preview_key=key
    controller._preview_base=controller.baseline.clone()
    controller._preview_authority=alpha.clone()
    controller._preview_clipped=clipped.clone()
    # All state terms and the baseline are visible; the actor can interpret the
    # gate and the previous FINAL command, not merely its own residual action.
    value=torch.cat((base_observation(env),dx,
                      (controller.baseline-controller._offset)/.25,alpha[:,None],
                      (controller.last_command-controller._offset)/.25,
                      controller.applied_saturation[:,None]),-1)
    env._lqr_observation_key,env._lqr_observation_value=key,value
    return value


def final_action_change(env):
    controller = term(env)
    return ((controller.last_command-controller.previous_command)/.25).square().mean(-1)


def metric(env, name):
    controller = term(env)
    if name == 'authority':
        return controller.applied_alpha
    if name == 'saturation':
        return controller.applied_saturation
    if name == 'residual_rms':
        scale=(controller.cfg.residual_scale+(1.-controller.applied_alpha)*
               (controller.cfg.fallback_scale-controller.cfg.residual_scale))[:,None]
        return (controller.raw_action.clamp(-2,2)*scale).square().mean(-1).sqrt()
    raise ValueError(name)
