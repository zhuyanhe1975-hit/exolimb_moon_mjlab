"""V3 native-sensor balance task; preserves legacy v2 and never trains on import."""
from copy import deepcopy
import math
import numpy as np
import mujoco
import torch
from stand_task import stand_spec,stand_skin_spec,GRAVITY
from terrain_stand_task import terrain_stand_cfg as legacy_cfg,reset_on_terrain,configure_terrain_view
from terrain_stand_task import terrain_stand_rl as legacy_rl
from lunar_terrain import surface,terrain_data
from balance_v3_mdp import SENSORS,OBS_DIM,observation,signal,alive
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Balance-v3'
EXPERIMENT='wangyu_moon_stand_balance_v3'
SPAWN_MARGIN=20.
STATION_RADIUS=1.
SPAWN_JITTER=1.25
SPAWN_CLEARANCE=.015
REWARD_WEIGHTS={'alive':.20,'pelvis_velocity':-4.,'centroidal_angular_momentum':-.002,
                'gravity_tilt':-2.,'action_change':-.002}


def add_balance_sensors(spec):
    spec.body('Pelvis').add_site(name='balance_root',size=[.01],rgba=[0,0,0,0],group=4)
    spec.body('Torso').add_site(name='balance_torso',size=[.01],rgba=[0,0,0,0],group=4)
    for side in ('Left','Right'):
        sole=spec.geom(side.lower()+'_sole')
        pos=np.array(sole.pos);pos[2]-=sole.size[2]
        # Native TOUCH measures normal contact force for the sole body. Sites
        # are sensor volumes only: no mass, collision or added joint constraint.
        spec.body(side+'Foot').add_site(name='balance_'+side.lower()+'_sole',
            type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,
            size=[float(sole.size[0])+.005,float(sole.size[1])+.005,.015],
            rgba=[0,0,0,0],group=4)
    for name,kind,objkind,objname,_ in SENSORS:
        spec.add_sensor(name='balance_'+name,type=getattr(mujoco.mjtSensor,'mjSENS_'+kind),
                        objtype=getattr(mujoco.mjtObj,'mjOBJ_'+objkind),objname=objname)
    return spec


def robot_spec():
    return add_balance_sensors(stand_spec())


def skin_spec():
    return add_balance_sensors(stand_skin_spec())


def configure_balance_view(spec):
    configure_terrain_view(spec)
    spec.visual.quality.shadowsize=4096
    spec.visual.map.shadowclip=1.5
    # Keep a little camera fill, without washing out the cast shadow.
    spec.visual.headlight.ambient=[.12,.12,.12]
    spec.visual.headlight.diffuse=[.18,.18,.18]
    for light in spec.lights:
        light.active=False
    # Native TRACK keeps the sun's world direction stable while its shadow
    # volume follows the tracked body across random 2km map locations.
    spec.body('robot/Pelvis').add_light(name='balance_sun',
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
        pos=[-3.,-4.,6.],dir=[.45,.55,-1.],castshadow=True,
        diffuse=[.85,.82,.76],ambient=[.06,.06,.06],specular=[.25,.25,.25])


def verify_native_terrain(env):
    """One-time CPU metadata audit of the ACTUAL compiled collision surface."""
    model=env.sim.mj_model;meta,normalized,spawn=terrain_data()
    geom=model.geom('site20_surface');gid=geom.id;field=int(model.geom_dataid[gid])
    if int(model.geom_type[gid])!=int(mujoco.mjtGeom.mjGEOM_HFIELD) or field<0:
        raise RuntimeError('Site20 must be an active native collision heightfield')
    body=int(model.geom_bodyid[gid])
    if int(model.body_parentid[body])!=0:
        raise RuntimeError('Unexpected terrain hierarchy; world-height lookup needs recalibration')
    np.testing.assert_allclose(model.body_pos[body],[0,0,0],atol=1e-9)
    np.testing.assert_allclose(model.body_quat[body],[1,0,0,0],atol=1e-9)
    np.testing.assert_allclose(model.geom_pos[gid],[0,0,meta['zmin_m']],atol=1e-6)
    np.testing.assert_allclose(model.geom_quat[gid],[1,0,0,0],atol=1e-9)
    np.testing.assert_allclose(model.hfield_size[field,:3],
        [meta['half_extent_m'],meta['half_extent_m'],meta['height_range_m']],atol=1e-6)
    if (int(model.hfield_nrow[field]),int(model.hfield_ncol[field]))!=normalized.shape:
        raise RuntimeError('Height sampler grid does not match the collider')
    address=int(model.hfield_adr[field]);count=normalized.size
    np.testing.assert_allclose(model.hfield_data[address:address+count],normalized.ravel(),atol=2e-6,rtol=0)
    for name in ('robot/left_sole','robot/right_sole'):
        foot=model.geom(name).id
        pair_enabled=((int(model.geom_contype[foot]) & int(model.geom_conaffinity[gid])) or
                      (int(model.geom_contype[gid]) & int(model.geom_conaffinity[foot])))
        if not pair_enabled: raise RuntimeError(f'Foot-ground collision disabled: {name}')
    ground=surface(env)
    # Filter once on CPU; per-reset sampling and sole checks are batched on GPU.
    keep=(np.abs(spawn)+SPAWN_JITTER<=meta['half_extent_m']-SPAWN_MARGIN).all(-1)
    if not keep.any(): raise RuntimeError('No valid spawn pool inside the finite terrain')
    ground.spawn_xy=torch.tensor(spawn[keep],device=env.device,dtype=torch.float32)
    env._balance_terrain_verified=True


def reset_balance(env,env_ids):
    if env_ids is not None and len(env_ids)==0: return
    if not getattr(env,'_balance_terrain_verified',False): verify_native_terrain(env)
    # Match the established Apollo terrain convention: +/-1.25m around a valid
    # anchor, full random yaw, rotated sole clearance of 15mm, zero reset velocity.
    reset_on_terrain(env,env_ids,position_jitter=SPAWN_JITTER,clearance=SPAWN_CLEARANCE)
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    root=env._stand_target_root[ids];heading=env._stand_target_heading[ids]
    c=heading[:,0].square()-heading[:,3].square();s=2*heading[:,0]*heading[:,3]
    ground=surface(env);foot=ground.sole_points
    xy=torch.stack((root[:,None,0]+c[:,None]*foot[None,:,0]-s[:,None]*foot[None,:,1],
                    root[:,None,1]+s[:,None]*foot[None,:,0]+c[:,None]*foot[None,:,1]),-1)
    gaps=root[:,None,2]+foot[None,:,2]-ground.height(xy)
    inside=(xy.abs()<ground.half-2).all(dim=(1,2))
    valid=(inside & torch.isfinite(root).all(-1) & torch.isfinite(gaps).all(-1)
           & (gaps.min(-1).values>=SPAWN_CLEARANCE-.0005) & (gaps.max(-1).values<=.20))
    # Deliberate single scalar read per RESET BATCH for a clear fail-fast error.
    # No per-world Python loops and no asynchronous device assertion that poisons CUDA.
    if not bool(valid.all()):
        bad=ids[~valid][:16].detach().cpu().tolist()
        raise RuntimeError(f'Illegal terrain spawn in worlds {bad}: bounds, finite values, penetration or >20cm air gap')
    if not getattr(env,'_balance_spawn_reported',False):
        stats=torch.stack((valid.sum(),gaps.min(),gaps.max(),
                           root[:,0].min(),root[:,0].max(),root[:,1].min(),root[:,1].max())).detach().cpu().tolist()
        print(f'[Spawn verified] {int(stats[0])}/{len(ids)} worlds; native hfield aligned; '
              f'sole gap={stats[1]:.4f}..{stats[2]:.4f}m; '
              f'X={stats[3]:.1f}..{stats[4]:.1f}m Y={stats[5]:.1f}..{stats[6]:.1f}m; '
              f'map margin>={SPAWN_MARGIN:g}m; station radius={STATION_RADIUS:g}m',flush=True)
        env._balance_spawn_reported=True


def left_station(env):
    if not hasattr(env,'_stand_target_root'):
        return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    p=env.scene['robot'].data.root_link_pos_w[:,:2]-env._stand_target_root[:,:2]
    return p.square().sum(-1)>STATION_RADIUS**2


def torso_tilted(env):
    # Uses current native site quaternion directly, not observation/reward caches.
    if not hasattr(env,'_balance_torso_quat_address'):
        env._balance_torso_quat_address=int(env.sim.mj_model.sensor('robot/balance_torso_quat').adr[0])
    address=env._balance_torso_quat_address
    q=env.sim.data.sensordata[:,address:address+4]
    return 1-2*(q[:,1].square()+q[:,2].square())<math.cos(math.radians(45.))


def terrain_stand_cfg(play=False):
    c=legacy_cfg(play=play)
    c.scene.entities['robot'].spec_fn=skin_spec if play else robot_spec
    c.events={'reset_pose':EventTermCfg(func=reset_balance,mode='reset')}
    c.observations={group:ObservationGroupCfg({'state':ObservationTermCfg(func=observation)},enable_corruption=False)
                    for group in ('actor','critic')}
    # More corrective authority than legacy +/-0.15rad, still within physical limits.
    physical=stand_spec().compile()
    clip={physical.joint(i).name:(max(-.5,float(physical.jnt_range[i,0])),min(.5,float(physical.jnt_range[i,1])))
          for i in range(1,physical.njnt)}
    c.actions={'position':JointPositionActionCfg(entity_name='robot',actuator_names=('.*',),scale=.25,clip=clip)}
    terms={'alive':(alive,{}),'pelvis_velocity':(signal,{'name':'pelvis_speed_squared'}),
           'centroidal_angular_momentum':(signal,{'name':'momentum_squared'}),
           'gravity_tilt':(signal,{'name':'tilt_squared'}),'action_change':(signal,{'name':'action_change'})}
    c.rewards={name:RewardTermCfg(func=fn,params=params,weight=REWARD_WEIGHTS[name])
               for name,(fn,params) in terms.items()}
    c.metrics={name:MetricsTermCfg(func=signal,params={'name':key},reduce='mean') for name,key in
        [('pelvis_speed_m_s','pelvis_speed'),('com_speed_m_s','com_speed'),
         ('centroidal_angular_momentum_kg_m2_s','momentum_norm'),
         ('gravity_tilt_mean_deg','root_tilt_deg'),('torso_tilt_mean_deg','torso_tilt_deg'),
         ('predicted_support_offset_m','predicted_support_offset'),('double_contact_fraction','double_contact')]}
    c.metrics['gravity_tilt_peak_deg']=MetricsTermCfg(func=signal,params={'name':'root_tilt_deg'},reduce='max')
    c.terminations['left_station']=TerminationTermCfg(func=left_station)
    c.terminations['torso_tilted']=TerminationTermCfg(func=torso_tilted)
    c.scene.num_envs=1 if play else 8192
    if play:
        c.viewer.enable_shadows=True
        c.scene.spec_fn=configure_balance_view
    return c


terrain_stand_rl=deepcopy(legacy_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
terrain_stand_rl.actor.hidden_dims=(128,128)
terrain_stand_rl.critic.hidden_dims=(128,128)
terrain_stand_rl.actor.obs_normalization=True
terrain_stand_rl.critic.obs_normalization=True
terrain_stand_rl.actor.distribution_cfg['init_std']=.25
terrain_stand_rl.clip_actions=2.
terrain_stand_rl.algorithm.gamma=.995
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=StandRunner)
