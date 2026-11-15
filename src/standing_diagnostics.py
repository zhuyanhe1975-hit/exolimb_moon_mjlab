"""Fail before invalid transitions reach PPO; diagnostic-only, no state repair.

MuJoCo's installed NanGuard detects nonfinite native state but does not stop
training and does not detect finite joint explosions. Reuse its detector, add
liberal physical sanity bounds, and dump only on failure (no rolling GPU clones).
"""
from datetime import datetime
from pathlib import Path
import os
import mujoco
import torch
from mjlab.utils.nan_guard import NanGuard


def anomaly_reasons(qpos,qvel,limits,nonfinite,qacc=None):
    """Floating root + 23 direct hinges. Limits are native compiled hard ranges.

    1rad limit overrun / 200rad/s are gross anomaly thresholds, not training
    targets or a fall detector. Soft constraint excursions are allowed.
    """
    joint=qpos[:,7:]
    over=((joint<limits[:,0]-1.)|(joint>limits[:,1]+1.)).any(-1)
    root_speed=qvel[:,:3].norm(dim=-1)>50.
    angular=qvel[:,3:].abs().amax(-1)>200.
    quaternion=qpos[:,3:7].norm(dim=-1)
    bad_quat=(quaternion<.5)|(quaternion>1.5)
    acceleration=torch.zeros_like(nonfinite) if qacc is None else qacc.abs().amax(-1)>1e8
    return torch.stack((nonfinite,over,root_speed,angular,bad_quat,acceleration,torch.zeros_like(nonfinite)),-1)


REASONS=('native_nan_or_inf','joint_range_overrun_1rad','root_speed_over_50m_s',
         'angular_speed_over_200rad_s','invalid_root_quaternion','native_acceleration_over_1e8','nonfinite_or_gross_observation')


def dump_and_raise(env,reasons,stage,extra=None):
    ids=reasons.any(-1).nonzero(as_tuple=False).flatten()
    selected=ids[:8]
    tiled=any(env.sim.mj_model.geom(i).name.endswith('_sole_tile_1') for i in range(env.sim.mj_model.ngeom))
    plane=any(int(kind)==int(mujoco.mjtGeom.mjGEOM_PLANE) for kind in env.sim.mj_model.geom_type)
    scene_cfg=getattr(getattr(env,'cfg',None),'scene',None)
    callback=getattr(scene_cfg,'spec_fn',None)
    local_dem=getattr(callback,'__module__','')=='lunar_finetune_task'
    actor=getattr(getattr(env,'cfg',None),'observations',{}).get('actor')
    hop=actor is not None and actor.terms['state'].func.__module__=='jump_mdp'
    if hop:
        experiment='hop_lunar_v1'
    elif local_dem:
        experiment='stand_lunar_v15'
    elif plane:
        groups=getattr(getattr(env,'cfg',None),'observations',{})
        actor=groups.get('actor')
        dimension=actor.terms['state'].params.get('obs_dim') if actor is not None else None
        if actor is not None and actor.terms['state'].func.__module__=='no_margin_mdp':dimension=143
        if actor is not None and actor.terms['state'].func.__module__=='torso_observation_mdp':dimension=146
        experiment='stand_flat_v14' if dimension==146 else ('stand_flat_v13' if dimension==143 else 'stand_support_flat_v12')
    else:
        experiment='stand_support_contact_v11' if tiled else 'stand_support_v10'
    directory=Path(os.environ.get('WANGYU_DIAGNOSTIC_DIR',
        str(Path(__file__).resolve().parents[1]/'outputs'/experiment/'anomalies')))
    directory.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path=directory/f'anomaly_{stamp}_{os.getpid()}.pt'
    sim=env.sim.data
    payload={'collision_asset':'four_tile_sole' if tiled else 'single_box_sole','stage':stage,'step':env.common_step_counter,'env_ids':ids.cpu(),
             'sampled_env_ids':selected.cpu(),'reason_names':REASONS,
             'reasons':reasons[selected].cpu(),'step_dt':env.step_dt,
             'note':'Unmodified terminal native state; no clipping/reset/optimizer update for this step.'}
    for name in ('qpos','qvel','qacc','qacc_warmstart','ctrl','sensordata','xfrc_applied'):
        payload[name]=getattr(sim,name)[selected].detach().cpu()
    for name in ('geom_dataid','geom_pos'):
        field=getattr(env.sim.model,name,None) if hasattr(env.sim,'model') else None
        if field is not None:
            payload['model_'+name]=(field[:1].expand((len(selected),)+tuple(field.shape[1:])) if field.shape[0]==1 else field[selected]).detach().cpu()
    if hasattr(env,'_local_lunar_surface'):
        payload['local_patch_ids']=env._local_lunar_surface.patch_ids[selected].cpu()
    payload['action']=env.action_manager.action[selected].detach().cpu()
    payload['previous_action']=env.action_manager.prev_action[selected].detach().cpu()
    payload['episode_length']=env.episode_length_buf[selected].cpu()
    payload['observation_input']={k:v[selected].detach().cpu() for k,v in env.obs_buf.items() if isinstance(v,torch.Tensor)}
    payload['joint_ranges']=env._standing_native_limits.cpu()
    payload['joint_names']=[env.sim.mj_model.joint(i).name for i in range(1,env.sim.mj_model.njnt)]
    if extra is not None:payload['invalid_observation']=extra[selected].detach().cpu()
    try:
        contact=env.scene['support_contacts'].data
        payload['contacts']={name:getattr(contact,name)[selected].detach().cpu()
                             for name in ('found','force','dist','pos')}
    except (KeyError,RuntimeError,AttributeError) as error:
        payload['contact_dump_error']=str(error)
    torch.save(payload,path)
    mujoco.mj_saveModel(env.sim.mj_model,str(path.with_suffix('.mjb')),None)
    raise RuntimeError(f'Standing numerical anomaly at step {env.common_step_counter}, stage={stage}, '
                       f'{ids.numel()} worlds. Raw state saved to {path}. '
                       'Stopped before this transition reaches PPO; rewards were not clipped.')


def state_reasons(env):
    """Build the reason mask on device, without reducing it to the host."""
    native=env.sim.data
    if not hasattr(env,'_standing_native_limits'):
        env._standing_native_limits=torch.tensor(env.sim.mj_model.jnt_range[1:].copy(),device=env.device,dtype=native.qpos.dtype)
        if native.qpos.shape[1]!=30 or native.qvel.shape[1]!=29:
            raise ValueError('Standing diagnostic guard requires the 23-hinge floating-root model')
    return anomaly_reasons(native.qpos,native.qvel,env._standing_native_limits,
                           NanGuard.detect_nans(native),native.qacc)


def check_state(env,stage,defer=False):
    """Fail-fast state guard. ``defer`` returns the mask so a caller can fuse
    its own validity bits into a single device-to-host reduction."""
    key=(env.common_step_counter,getattr(env,'_stand_reset_generation',0),stage)
    if getattr(env,'_standing_guard_key',None)==key:return None
    reasons=state_reasons(env)
    if defer:return reasons
    if reasons.any():dump_and_raise(env,reasons,stage)
    env._standing_guard_key=key
    return None


def overflow_guard(env):
    """Accumulate MJWarp's per-world overflow bitmask on device; fail per rollout.

    MJWarp clears a world's bitmask in reset_data, so it is OR-accumulated here
    before auto-reset (pre_reward). The accumulation is one device op per step;
    the host check runs every WANGYU_OVERFLOW_CHECK_STEPS control steps (one
    rollout), per the MJWarp guidance to avoid a sync every simulation step.
    """
    overflow=getattr(env.sim.data,'overflow',None)
    if overflow is None:return
    if not hasattr(env,'_standing_overflow'):
        env._standing_overflow=torch.zeros(env.num_envs,dtype=torch.int32,device=env.device)
        env._standing_overflow_every=max(1,int(os.environ.get('WANGYU_OVERFLOW_CHECK_STEPS','32')))
    torch.bitwise_or(env._standing_overflow,overflow,out=env._standing_overflow)
    if env.common_step_counter%env._standing_overflow_every:return
    if bool(env._standing_overflow.any()):
        from mujoco_warp._src.types import OverflowType
        bits=0
        for value in env._standing_overflow.unique().cpu().tolist():bits|=int(value)
        names=[f.name for f in OverflowType if bits&int(f.value)]
        worlds=(env._standing_overflow!=0).nonzero(as_tuple=False).flatten()
        cfg=getattr(getattr(env,'cfg',None),'sim',None)
        raise RuntimeError(
            f'MJWarp capacity overflow {names} in {worlds.numel()} worlds '
            f'(first {worlds[:8].cpu().tolist()}) by step {env.common_step_counter}; '
            f'nconmax={getattr(cfg,"nconmax",None)} njmax={getattr(cfg,"njmax",None)} '
            f'contact_sensor_maxmatch={getattr(cfg,"contact_sensor_maxmatch",None)}. '
            'Overflowed contacts/constraints were dropped, so these transitions are '
            'not valid physics; raise the capacity instead of continuing.')


def numerical_guard(env):
    # First termination term: fail before reward/metrics read an exploded state.
    check_state(env,'pre_reward')
    overflow_guard(env)
    return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)


def guarded_observation(env,observation_fn=None,obs_dim=None):
    if observation_fn is None:
        from support_mdp import observation as observation_fn,OBS_DIM
        if obs_dim is None:obs_dim=OBS_DIM
    if obs_dim is None:raise ValueError('Custom guarded observation requires obs_dim')
    # MJLab calls terms while constructing ObservationManager to infer shape,
    # before the terrain reset. This is not an actual policy observation.
    if not hasattr(env,'observation_manager'):
        return torch.zeros((env.num_envs,obs_dim),device=env.device)
    key=(env.common_step_counter,getattr(env,'_stand_reset_generation',0),observation_fn,obs_dim)
    if getattr(env,'_standing_checked_obs_key',None)==key:
        return env._standing_checked_obs
    # The native-state mask and the observation validity bits are reduced to the
    # host together: one synchronization per control step instead of two. The
    # thresholds, reason bits, raised stage and saved evidence are unchanged; a
    # native anomaly still reports 'post_forward' and still dumps raw state.
    state_key=(env.common_step_counter,getattr(env,'_stand_reset_generation',0),'post_forward')
    reasons=check_state(env,'post_forward',defer=True)
    result=observation_fn(env)
    if result.shape!=(env.num_envs,obs_dim):
        raise ValueError(f'Observation shape {tuple(result.shape)} != {(env.num_envs,obs_dim)}')
    invalid=(~torch.isfinite(result).all(-1))|(result.abs().amax(-1)>1e5)
    if reasons is None:
        reasons=torch.zeros((env.num_envs,len(REASONS)),device=env.device,dtype=torch.bool)
    reasons[:,-1]=invalid
    if reasons.any():
        # Native state anomalies take priority: report the physics stage, not
        # the observation that merely inherited the invalid values.
        stage='post_forward' if bool(reasons[:,:-1].any()) else 'observation'
        dump_and_raise(env,reasons,stage,result)
    env._standing_guard_key=state_key
    env._standing_checked_obs_key=key
    env._standing_checked_obs=result
    return result


def joint_speed(env):
    return env.scene['robot'].data.joint_vel.square().mean(-1).sqrt()


def root_angular_speed(env):
    return env.scene['robot'].data.root_link_ang_vel_b.norm(dim=-1)


def settled_mask(env):
    """No native external force/torque for >=2s; newborns also wait >=2s."""
    key=(env.common_step_counter,getattr(env,'_stand_reset_generation',0))
    if getattr(env,'_standing_quiet_key',None)!=key:
        if not hasattr(env,'_standing_last_wrench'):
            env._standing_last_wrench=torch.full((env.num_envs,),env.common_step_counter,device=env.device,dtype=torch.long)
        wrench=env.scene['robot'].data.body_external_wrench
        active=wrench.abs().flatten(1).amax(-1)>1e-6
        active|=env.sim.data.qfrc_applied.abs().amax(-1)>1e-6
        newborn=env.episode_length_buf<=1
        env._standing_last_wrench.masked_fill_(active|newborn,env.common_step_counter)
        env._standing_settled=(env.common_step_counter-env._standing_last_wrench)*env.step_dt>=2.
        env._standing_quiet_key=key
    return env._standing_settled


def settled_fraction(env):
    return settled_mask(env).float()


def moving_without_wrench(env):
    robot=env.scene['robot'].data
    moving=(joint_speed(env)>.1)|(robot.root_link_lin_vel_b.norm(dim=-1)>.03)|(root_angular_speed(env)>.1)
    # Fraction of all sampled time, restricted to settled intervals. Divide by
    # settled_unforced_fraction for the conditional fraction, with matched windows.
    return (settled_mask(env)&moving).float()


def contact_count(env):
    contact=env.scene['support_contacts'].data
    valid=(contact.found>0)&(contact.force.norm(dim=-1)>.5)&(contact.dist<=.002)
    return valid.sum(-1).float()
