"""v12: no-LQR standing on a native infinite plane; original single-box soles."""
from copy import deepcopy
from functools import lru_cache
import os
import mujoco
import numpy as np
import torch
import balance_task as physical
import support_task as base
import torso_velocity
from stand_compute import invalidate
from mjlab.envs.mdp.events import reset_scene_to_default,resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactSensorCfg,ContactMatch
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Support-Flat-v12'
EXPERIMENT=os.environ.get('WANGYU_TRAIN_EXPERIMENT','wangyu_moon_stand_support_flat_v12_contactsync1_quiet1_torso2')
OUTPUT_SUBDIR='stand_support_flat_v12'
IS_FLAT=True
CONTACT_SYNC_REVISION='contactsync1'
OBS_DIM=base.OBS_DIM
SPAWN_MARGIN=0.
STATION_RADIUS=base.STATION_RADIUS
PUSH_PARAMS=base.PUSH_PARAMS
REWARD_WEIGHTS={name:weight for name,weight in base.REWARD_WEIGHTS.items() if name!='pelvis_velocity'}
REWARD_WEIGHTS.update(torso_velocity=-4.,torso_angular_velocity=-.2,centroidal_angular_momentum=-.01)
CLEARANCE=.015


@lru_cache(maxsize=1)
def rest_height():
    # Static native kinematics/distance query only: no mj_forward/mj_step.
    spec=physical.robot_spec()
    spec.worldbody.add_geom(name='rest_floor_probe',type=mujoco.mjtGeom.mjGEOM_PLANE,
                           size=[1,1,.01],contype=0,conaffinity=0)
    model=spec.compile();data=mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    plane=model.geom('rest_floor_probe').id
    distance=min(mujoco.mj_geomDistance(model,data,plane,model.geom(side+'_sole').id,2.,None)
                 for side in ('left','right'))
    if not np.isfinite(distance) or abs(distance)>.5:
        raise ValueError(f'Unexpected source-pose sole gap: {distance}')
    return float(model.qpos0[2]-distance+CLEARANCE)


def configure_plane(spec,play=False):
    planes=[g for g in spec.geoms if g.type==mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes)!=1 or len(list(spec.hfields))!=0:
        raise ValueError('Flat standing requires exactly one native plane and no heightfields')
    plane=planes[0];plane.pos=[0,0,0]
    plane.contype=1;plane.conaffinity=1;plane.friction=[1.,.01,.001]
    plane.rgba=[.32,.34,.37,1.]
    if play:physical.configure_balance_view(spec)


def configure_play_plane(spec):
    configure_plane(spec,play=True)


def reset_plane(env,env_ids):
    ids=resolve_env_ids(env,env_ids)
    if len(ids)==0:return
    reset_scene_to_default(env,ids)
    if not getattr(env,'_flat_stand_verified',False):
        model=env.sim.mj_model
        planes=np.flatnonzero(model.geom_type==int(mujoco.mjtGeom.mjGEOM_PLANE))
        if len(planes)!=1 or model.nhfield!=0:
            raise ValueError('Expected native infinite plane; heightfields must be absent')
        np.testing.assert_allclose(model.geom_pos[int(planes[0])],[0,0,0],atol=1e-9)
        env._flat_stand_verified=True
        print(f'[Flat stand] native infinite plane; source joint pose; zero velocity; '
              f'sole gap={CLEARANCE:g}m; no LQR, no DEM',flush=True)
    robot=env.scene['robot'];d=robot.data
    state=d.default_root_state[ids].clone()
    state[:,:3]+=env.scene.env_origins[ids]
    state[:,2]=rest_height();state[:,3:7]=0.;state[:,3]=1.;state[:,7:]=0.
    robot.write_root_state_to_sim(state,env_ids=ids)
    joint=d.default_joint_pos[ids]
    robot.write_joint_state_to_sim(joint,torch.zeros_like(joint),env_ids=ids)
    robot.set_joint_position_target(joint,env_ids=ids)
    if not hasattr(env,'_stand_target_root'):
        env._stand_target_root=torch.zeros((env.num_envs,3),device=env.device)
        env._stand_target_heading=torch.zeros((env.num_envs,4),device=env.device)
    env._stand_target_root[ids]=state[:,:3]
    env._stand_target_heading[ids]=state[:,3:7]
    invalidate(env,ids)


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    cfg.episode_length_s=10.
    cfg.scene.entities['robot'].spec_fn=torso_velocity.skin_spec if play else torso_velocity.robot_spec
    cfg.rewards.pop('pelvis_velocity')
    cfg.rewards['torso_velocity']=RewardTermCfg(func=torso_velocity.squared,weight=REWARD_WEIGHTS['torso_velocity'])
    cfg.rewards['torso_angular_velocity']=RewardTermCfg(func=torso_velocity.angular_squared,weight=REWARD_WEIGHTS['torso_angular_velocity'])
    cfg.metrics['torso_angular_speed_rad_s']=MetricsTermCfg(func=torso_velocity.angular_speed,reduce='mean')
    cfg.metrics['torso_speed_m_s']=MetricsTermCfg(func=torso_velocity.speed,reduce='mean')
    for name,term in cfg.rewards.items():term.weight=REWARD_WEIGHTS[name]
    cfg.scene.terrain=TerrainEntityCfg(terrain_type='plane',env_spacing=3.,textures=(),materials=())
    cfg.scene.spec_fn=configure_play_plane if play else configure_plane
    cfg.scene.sensors=tuple(ContactSensorCfg(name='support_contacts',
        primary=ContactMatch(mode='geom',pattern=('left_sole','right_sole'),entity='robot'),
        secondary=ContactMatch(mode='geom',pattern='terrain'),
        fields=('found','force','dist','pos'),reduce='maxforce',num_slots=4)
        if s.name=='support_contacts' else s for s in cfg.scene.sensors)
    cfg.events['reset_pose']=EventTermCfg(func=reset_plane,mode='reset')
    cfg.terminations.pop('terrain_edge',None)
    rest_height()  # Validate/cache source-pose clearance with native geometry.
    return cfg


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=StandRunner)
