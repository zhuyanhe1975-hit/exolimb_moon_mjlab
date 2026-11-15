"""Passive playback diagnostics; read native state and already-computed rewards."""
from pathlib import Path
import mujoco
import numpy as np
import torch


def joint_control_indices(model,joint_names):
    """Native controls are actuator ordered, not necessarily joint ordered."""
    result=[]
    for name in joint_names:
        joint=model.joint('robot/'+name).id
        matches=np.flatnonzero((model.actuator_trnid[:,0]==joint)&(model.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT)))
        if len(matches)!=1:
            raise ValueError(f'Expected one direct joint actuator for {name}, got {len(matches)}')
        result.append(int(matches[0]))
    return result


class MotionDiagnostics:
    def __init__(self,env):
        self.env=env;self.names=('pelvis_linear_m_s','pelvis_angular_rad_s','com_linear_m_s',
            'centroidal_momentum_kg_m2_s','joint_speed_rad_s','target_error_rad','target_change_rad_per_step')
        model=env.sim.mj_model;indices=[]
        for name in ('root_vel','root_omega','com_vel','momentum'):
            sensor=model.sensor('robot/balance_'+name);a=int(sensor.adr[0]);indices.extend(range(a,a+3))
        self.indices=torch.tensor(indices,device=env.device)
        self.control_indices=torch.tensor(joint_control_indices(model,env.scene['robot'].joint_names),device=env.device)
        torso_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,'robot/balance_torso_com_vel')
        self.torso_address=int(model.sensor_adr[torso_id]) if torso_id>=0 else None
        if self.torso_address is not None:self.names+=('torso_linear_m_s',)
        self.torso_trace=[]
        angular_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,'robot/balance_torso_omega')
        self.torso_angular_address=int(model.sensor_adr[angular_id]) if angular_id>=0 else None
        if self.torso_angular_address is not None:self.names+=('torso_angular_rad_s',)
        self.torso_angular_trace=[]
        self.terms=list(env.reward_manager.active_terms)
        self.weights={name:env.reward_manager.get_term_cfg(name).weight for name in self.terms}
        self.sums=torch.zeros(len(self.names),device=env.device)
        self.rates=torch.zeros(len(self.terms),device=env.device)
        self.count=torch.zeros((),device=env.device)
        self.previous_target=None;self.trace=[]

    def record(self,quiet,done):
        env=self.env;robot=env.scene['robot'].data
        native=env.sim.data.sensordata[:,self.indices]
        target=env.sim.data.ctrl[:,self.control_indices]
        delta=torch.zeros_like(target) if self.previous_target is None else target-self.previous_target
        error=target-robot.joint_pos
        magnitude=torch.stack((native[:,:3].norm(dim=-1),native[:,3:6].norm(dim=-1),
            native[:,6:9].norm(dim=-1),native[:,9:12].norm(dim=-1),
            robot.joint_vel.square().mean(-1).sqrt(),error.square().mean(-1).sqrt(),
            delta.square().mean(-1).sqrt()),-1)
        if self.torso_address is not None:
            torso=env.sim.data.sensordata[:,self.torso_address:self.torso_address+3]
            magnitude=torch.cat((magnitude,torso.norm(dim=-1)[:,None]),-1)
            self.torso_trace.append(torso[0].clone())
        if self.torso_angular_address is not None:
            omega=env.sim.data.sensordata[:,self.torso_angular_address:self.torso_angular_address+3]
            magnitude=torch.cat((magnitude,omega.norm(dim=-1)[:,None]),-1)
            self.torso_angular_trace.append(omega[0].clone())
        # Installed RewardManager exposes only a per-world CPU debug iterator.
        # Its internal tensor is the already-computed weighted reward RATE
        # (compute() stores value/scale here). Read-only; never recompute rewards.
        rate=env.reward_manager._step_reward
        self.sums+=(magnitude.square()*quiet[:,None]).sum(0)
        self.rates+=(rate*quiet[:,None]).sum(0);self.count+=quiet.sum()
        self.trace.append(torch.cat((native[0],robot.joint_pos[0],robot.joint_vel[0],target[0],
            rate[0],quiet[:1].float(),done[:1].float())).clone())
        self.previous_target=target.clone()

    def report(self,path):
        path=Path(path).with_suffix('.motion.npz');path.parent.mkdir(parents=True,exist_ok=True)
        env=self.env;trace=torch.stack(self.trace).cpu().numpy()
        n=env.scene['robot'].data.joint_pos.shape[1];k=len(self.terms)
        extra={'torso_velocity_world':torch.stack(self.torso_trace).cpu().numpy()} if self.torso_trace else {}
        if self.torso_angular_trace:extra['torso_omega_world']=torch.stack(self.torso_angular_trace).cpu().numpy()
        np.savez_compressed(path,time_s=(np.arange(len(trace))+1)*env.step_dt,
            pelvis_velocity_world=trace[:,:3],pelvis_omega_world=trace[:,3:6],
            com_velocity_world=trace[:,6:9],centroidal_momentum_world=trace[:,9:12],
            joint_position=trace[:,12:12+n],joint_velocity=trace[:,12+n:12+2*n],
            joint_target=trace[:,12+2*n:12+3*n],reward_rate=trace[:,12+3*n:12+3*n+k],
            reward_terms=np.array(self.terms),settled_unforced=trace[:,-2].astype(bool),
            reset=trace[:,-1].astype(bool),
            joint_names=np.array(env.scene['robot'].joint_names),target_indexing=np.array('joint_names'),**extra)
        count=float(self.count)
        rms=(self.sums/max(count,1)).sqrt().cpu().tolist()
        rates=(self.rates/max(count,1)).cpu().tolist()
        return {'settled_unforced_samples':int(count),
            'rms':dict(zip(self.names,rms)) if count else None,
            'weighted_reward_rates':dict(zip(self.terms,rates)) if count else None,
            'reward_weights':self.weights,'representative_env_0_trace':str(path),
            'scope':'Deterministic playback. Native kinematics post-forward; native reward rates pre-forward (one physics-substep lag). Quiet mask excludes first 2s after reset/external wrench. No reward or dynamics modifications.'}
