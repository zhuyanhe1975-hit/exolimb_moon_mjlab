"""V7: flat LQR with invariant equilibrium feedforward and continuous authority."""
from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.envs.mdp.events import reset_scene_to_default,resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg

import lqr_task as parent
from lqr_action import term,ResidualLqrAction,ResidualLqrActionCfg
from lqr_runner import LqrRunner as BaseLqrRunner
from stand_compute import invalidate
from lqr_flat_control import equilibrium_commands

TASK_ID='Wangyu-Moon-Stand-LQR-Flat-v7'
EXPERIMENT='wangyu_moon_stand_lqr_flat_v7'
OBS_DIM=parent.OBS_DIM
REWARD_WEIGHTS=dict(parent.REWARD_WEIGHTS)
STATION_RADIUS=parent.STATION_RADIUS


class LqrRunner(BaseLqrRunner):
    def save(self,path,infos=None):
        return super().save(path,infos={**(infos or {}),'lqr_control_variant':'flat-equilibrium-v7'})

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('infos',{}).get('lqr_control_variant')!='flat-equilibrium-v7':
            raise ValueError('This policy was not trained with flat v7 control; use stand-v6 for old flat policies')
        return super().load(path,load_cfg=load_cfg,strict=strict,map_location=map_location)


class EquilibriumLqrAction(ResidualLqrAction):
    def preview(self,residual=None):
        # Keep native state collection/phase ownership identical to v5/v6.
        # Only the flat controller's feedforward/authority composition changes.
        dx,_,authority,_=super().preview()
        if residual is None:
            residual=torch.zeros_like(self.raw_action)
        base,final,authority,clipped=equilibrium_commands(dx,self.K,self.ctrl0,authority,residual,
            self._clip[:,:,0],self._clip[:,:,1],self.cfg.residual_scale,self.cfg.fallback_scale)
        self.baseline.copy_(base)
        self.preview_alpha.copy_(authority)
        return dx,final,authority,clipped

    def reset(self,env_ids=None):
        super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        for buffer in (self.last_command,self.previous_command,self._processed_actions,self.baseline):
            buffer[ids]=self.ctrl0


@dataclass(kw_only=True)
class EquilibriumLqrActionCfg(ResidualLqrActionCfg):
    def build(self,env):
        return EquilibriumLqrAction(self,env)


def configure_plane(spec,play=False):
    planes=[geom for geom in spec.geoms if geom.type==mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes)!=1:
        raise ValueError('Flat LQR requires exactly one native infinite plane')
    plane=planes[0]
    plane.pos=[0,0,0]
    plane.contype=1;plane.conaffinity=1
    plane.friction=[1.,.01,.001]
    plane.rgba=[.32,.34,.37,1.]
    if play:
        parent.base.configure_balance_view(spec)


def configure_play_plane(spec):
    configure_plane(spec,play=True)


def reset_equilibrium(env,env_ids):
    ids=resolve_env_ids(env,env_ids)
    if len(ids)==0:return
    reset_scene_to_default(env,ids)
    robot=env.scene['robot'];controller=term(env)
    if not getattr(env,'_flat_lqr_verified',False):
        model=env.sim.mj_model
        planes=np.flatnonzero(model.geom_type==int(mujoco.mjtGeom.mjGEOM_PLANE))
        if len(planes)!=1 or np.any(model.geom_type==int(mujoco.mjtGeom.mjGEOM_HFIELD)):
            raise ValueError('Flat task must use a plane, never an hfield')
        gid=int(planes[0])
        np.testing.assert_allclose(model.geom_pos[gid],[0,0,0],atol=1e-9)
        np.testing.assert_allclose(model.geom_friction[gid],[1.,.01,.001],atol=1e-9)
        env._flat_lqr_verified=True
        print('[Flat LQR] native plane; calibrated equilibrium reset; zero velocity; '
              'no reset gap, random pose or pushes',flush=True)
    qref=controller.qref
    state=robot.data.default_root_state[ids].clone()
    state[:,:3]=env.scene.env_origins[ids]+qref[:3]
    state[:,3:7]=qref[3:7]
    state[:,7:]=0.
    robot.write_root_state_to_sim(state,env_ids=ids)
    joint=qref[7:].expand(len(ids),-1)
    robot.write_joint_state_to_sim(joint,torch.zeros_like(joint),env_ids=ids)
    robot.set_joint_position_target(controller.ctrl0.expand(len(ids),-1),env_ids=ids)
    if not hasattr(env,'_stand_target_root'):
        env._stand_target_root=(env.scene.env_origins+qref[:3]).clone()
        env._stand_target_heading=robot.data.default_root_state[:,3:7].clone()
    env._stand_target_root[ids]=state[:,:3]
    # Reference quaternion may lean; heading is ONLY the world-yaw frame.
    env._stand_target_heading[ids]=0.
    env._stand_target_heading[ids,0]=1.
    invalidate(env,ids)


def terrain_stand_cfg(play=False):
    cfg=parent.terrain_stand_cfg(play=play)
    cfg.actions={'position':EquilibriumLqrActionCfg(entity_name='robot',actuator_names=('.*',),
        scale=1.,clip=cfg.actions['position'].clip,residual_scale=.05)}
    cfg.scene.terrain=TerrainEntityCfg(terrain_type='plane',env_spacing=3.,textures=(),materials=())
    cfg.scene.spec_fn=configure_play_plane if play else configure_plane
    cfg.events={'reset_pose':EventTermCfg(func=reset_equilibrium,mode='reset')}
    cfg.terminations.pop('terrain_edge',None)
    cfg.sim.nconmax=16
    cfg.sim.njmax=128
    return cfg


terrain_stand_rl=deepcopy(parent.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),
                    terrain_stand_rl,runner_cls=LqrRunner)
