"""Independent random-location/yaw Site20 standing task; no training on import."""
from copy import deepcopy
import math
import os
import mujoco
import torch
from stand_task import stand_cfg,stand_rl,stand_spec,stand_skin_spec,action_change
from stand_compute import bundle,invalidate
from stand_runner import StandRunner
from tensor_graph import FixedTensorGraph
from lunar_terrain import Site20TerrainCfg,surface
from mjlab.envs.mdp.events import reset_scene_to_default,resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.viewer import ViewerConfig

TASK_ID='Wangyu-Moon-Stand-Terrain-v2'
EXPERIMENT='wangyu_moon_stand_terrain_v2'
OBS_DIM=97


def _with_com_velocity(spec):
    # Pelvis is the floating root of all 13 human bodies; the static terrain
    # is not a descendant. This sensor enables native subtree velocity updates
    # in both MuJoCo and MuJoCo Warp (sensor_subtree_vel).
    spec.add_sensor(name='whole_body_com_velocity',type=mujoco.mjtSensor.mjSENS_SUBTREELINVEL,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,objname='Pelvis')
    return spec


def terrain_robot_spec():
    return _with_com_velocity(stand_spec())


def terrain_skin_spec():
    return _with_com_velocity(stand_skin_spec())


def configure_terrain_view(spec):
    """Display-only settings; near clip must fit a person, not a 2km DEM."""
    spec.visual.map.znear=.002
    spec.visual.map.zfar=1000.
    spec.visual.headlight.active=1
    spec.visual.headlight.ambient=[.35,.35,.35]
    spec.visual.headlight.diffuse=[.65,.65,.65]


def reset_on_terrain(env,env_ids,position_jitter=None,clearance=.01):
    ids=resolve_env_ids(env,env_ids)
    reset_scene_to_default(env,ids)
    robot=env.scene['robot'];data=robot.data;ground=surface(env)
    n=len(ids)
    indices=torch.randint(ground.spawn_xy.shape[0],(n,),device=env.device)
    jitter=ground.meta['spawn_jitter_m'] if position_jitter is None else position_jitter
    xy=ground.spawn_xy[indices]+(2*torch.rand((n,2),device=env.device)-1)*jitter
    yaw=(2*torch.rand(n,device=env.device)-1)*math.pi
    c,s=torch.cos(yaw),torch.sin(yaw)
    foot=ground.sole_points
    points=torch.stack((xy[:,None,0]+c[:,None]*foot[None,:,0]-s[:,None]*foot[None,:,1],
                        xy[:,None,1]+s[:,None]*foot[None,:,0]+c[:,None]*foot[None,:,1]),-1)
    # Lift all sole samples clear of their own rotated terrain footprint.
    # Two rigid flat soles cannot simultaneously be flush with an arbitrary slope.
    z=(ground.height(points)-foot[None,:,2]).max(-1).values+clearance
    state=data.default_root_state[ids].clone()
    state[:,:2]=xy;state[:,2]=z
    state[:,3:7]=0.;state[:,3]=torch.cos(yaw/2);state[:,6]=torch.sin(yaw/2)
    state[:,7:]=0.
    robot.write_root_state_to_sim(state,env_ids=ids)
    if not hasattr(env,'_stand_target_root'):
        env._stand_target_root=(data.default_root_state[:,:3]+env.scene.env_origins).clone()
        env._stand_target_heading=data.default_root_state[:,3:7].clone()
    env._stand_target_root[ids]=state[:,:3]
    env._stand_target_heading[ids]=state[:,3:7]
    # Keep origin markers aligned with each randomized environment.
    env.scene.env_origins[ids,:2]=xy
    env.scene.env_origins[ids,2]=ground.height(xy)
    invalidate(env,ids)


def observation(env):
    base=bundle(env)[:,:72]
    key=(env.common_step_counter,getattr(env,'_stand_reset_generation',0))
    if getattr(env,'_terrain_scan_key',None)!=key:
        ground=surface(env);root=env.scene['robot'].data.root_link_pose_w
        if root.is_cuda and os.environ.get('WANGYU_TENSOR_GRAPH','1')!='0':
            if not hasattr(env,'_terrain_scan_graph'):
                env._terrain_scan_graph=FixedTensorGraph(ground.scan,root)
            features=env._terrain_scan_graph(root)
        else: features=ground.scan(root)
        env._terrain_scan_key,env._terrain_scan=key,features
    return torch.cat((base,env._terrain_scan),-1)


def com_velocity(env):
    """Native whole-human COM linear velocity in world coordinates, m/s."""
    if not hasattr(env,'_stand_com_sensor_adr'):
        sensor=env.sim.mj_model.sensor('robot/whole_body_com_velocity')
        env._stand_com_sensor_adr=int(sensor.adr[0])
    address=env._stand_com_sensor_adr
    return env.sim.data.sensordata[:,address:address+3]


def stability(env,speed_std=.05):
    """Only total COM speed: 1 at rest, exp(-1) at speed_std m/s."""
    return torch.exp(-com_velocity(env).square().sum(-1)/speed_std**2)


def com_speed(env):
    return com_velocity(env).norm(dim=-1)


def gravity_tilt_squared(env):
    """Positive rad²; RewardTermCfg supplies the negative penalty weight."""
    return bundle(env)[:,74]


def fallen(env):
    root=env.scene['robot'].data.root_link_pos_w
    nominal=env.scene['robot'].data.default_root_state[:,2]
    return root[:,2]-surface(env).root_height(root[:,:2])<.70*nominal


def edge(env):
    return (env.scene['robot'].data.root_link_pos_w[:,:2].abs()>surface(env).half-2).any(-1)


def terrain_stand_cfg(play=False):
    c=stand_cfg(play=play)
    c.scene.entities['robot'].spec_fn=terrain_skin_spec if play else terrain_robot_spec
    c.scene.terrain=TerrainEntityCfg(terrain_type='generator',textures=(),materials=(),
        terrain_generator=TerrainGeneratorCfg(seed=42,size=(2000.,2000.),num_rows=1,num_cols=1,
            curriculum=False,border_width=0.,color_scheme='none',sub_terrains={'site20':Site20TerrainCfg()}))
    c.scene.num_envs=1 if play else 8192
    c.observations={group:ObservationGroupCfg({'state':ObservationTermCfg(func=observation)},enable_corruption=False)
                    for group in ('actor','critic')}
    c.events={'reset_pose':EventTermCfg(func=reset_on_terrain,mode='reset')}
    c.rewards={'stability':RewardTermCfg(func=stability,weight=1.,params={'speed_std':.05}),
               'gravity_tilt':RewardTermCfg(func=gravity_tilt_squared,weight=-1.),
               'action_change':RewardTermCfg(func=action_change,weight=-.002)}
    c.metrics['com_speed_m_s']=MetricsTermCfg(func=com_speed,reduce='mean')
    c.terminations['fallen']=TerminationTermCfg(func=fallen)
    c.terminations['terrain_edge']=TerminationTermCfg(func=edge)
    # Hfield prisms can produce more contacts than two planes. Reserve capacity,
    # without asserting the old 8-contact bound or truncating new constraints.
    c.sim.nconmax=128
    c.sim.njmax=512
    if play:
        # Same local visual extent used by MJLab's offscreen renderer. Auto
        # extent from the 2km heightfield otherwise clips the close-up subject.
        c.scene.extent=max(4.,1.5*c.viewer.distance)
        c.scene.spec_fn=configure_terrain_view
        c.viewer.origin_type=ViewerConfig.OriginType.ASSET_BODY
        c.viewer.entity_name='robot'
        c.viewer.body_name='Pelvis'
    return c


terrain_stand_rl=deepcopy(stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=StandRunner)
