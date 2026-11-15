"""v15: same 146-input v14 policy, random local Site20 terrain transfer."""
from copy import deepcopy
import os
import json
from pathlib import Path
import numpy as np
import torch
import flat_torso_observation_task as base
import balance_task
import local_lunar_terrain as terrain
from mjlab.envs.mdp.events import reset_scene_to_default,resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg,requires_model_fields
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from stand_compute import invalidate
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Lunar-v15'
EXPERIMENT=os.environ.get('WANGYU_TRAIN_EXPERIMENT','wangyu_moon_stand_lunar_v15')
OUTPUT_SUBDIR='stand_lunar_v15'
PHYSICS_REVISION=terrain.REVISION
IS_FLAT=False
SUPPORT_PROXY_ENABLED=False
TERRAIN_DESCRIPTION='Site20 original 2km DEM + synthetic small craters/rims/roughness; 64 local 16m collision patches, .1m sampling; random patch/XY/yaw'
OBSERVATION_DESCRIPTION=base.OBSERVATION_DESCRIPTION
OBS_DIM=base.OBS_DIM
SPAWN_MARGIN=5.
STATION_RADIUS=base.STATION_RADIUS
PUSH_PARAMS=base.PUSH_PARAMS
REWARD_WEIGHTS=dict(base.REWARD_WEIGHTS)


def configure_scene(spec):
    terrain.configure_native(spec)


def configure_view(spec):
    terrain.configure_native(spec,full_view=True);balance_task.configure_balance_view(spec)


@requires_model_fields('geom_dataid','geom_pos')
def reset_local(env,env_ids,distinct_patches=False):
    ids=resolve_env_ids(env,env_ids)
    if len(ids)==0:return
    reset_scene_to_default(env,ids)
    if not hasattr(env,'_local_lunar_surface'):
        env._local_lunar_surface=terrain.LocalSurface(env)
        env._lunar_surface=env._local_lunar_surface
    ground=env._local_lunar_surface
    if distinct_patches and env.num_envs<=len(ground.spawnable_ids):
        available=torch.ones(terrain.COUNT,dtype=torch.bool,device=env.device)
        available[ground.spawn_counts==0]=False
        others=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);others[ids]=False
        available[ground.patch_ids[others]]=False
        pool=available.nonzero(as_tuple=False).flatten()
        chosen=pool[torch.randperm(len(pool),device=env.device)[:len(ids)]]
    else:
        chosen=ground.spawnable_ids[torch.randint(len(ground.spawnable_ids),(len(ids),),device=env.device)]
    ground.patch_ids[ids]=chosen
    env.sim.model.geom_dataid[ids,ground.geom_id]=ground.native_ids[chosen]
    env.sim.model.geom_pos[ids,ground.geom_id,2]=ground.zmin[chosen]
    if ground.visual_id is not None:
        env.sim.model.geom_pos[ids[:,None],ground.visual_ids[None,:]]=ground.visual_offsets[chosen]
    placement=(torch.rand(len(ids),device=env.device)*ground.spawn_counts[chosen]).long()
    xy=ground.spawn_xy[chosen,placement]+(2*torch.rand(len(ids),2,device=env.device)-1)*.025
    yaw=(2*torch.rand(len(ids),device=env.device)-1)*torch.pi
    c,s=yaw.cos(),yaw.sin();foot=ground.sole_points
    points=torch.stack((xy[:,None,0]+c[:,None]*foot[None,:,0]-s[:,None]*foot[None,:,1],
                        xy[:,None,1]+s[:,None]*foot[None,:,0]+c[:,None]*foot[None,:,1]),-1)
    heights=ground.height(points,ids)
    certified=(points.abs()<=2).all()
    z=(heights-foot[None,:,2]).max(-1).values+terrain.CLEARANCE+ground.clearance_padding[chosen]
    robot=env.scene['robot'];state=robot.data.default_root_state[ids].clone()
    state[:,:2]=xy;state[:,2]=z;state[:,3:7]=0
    state[:,3]=(yaw/2).cos();state[:,6]=(yaw/2).sin();state[:,7:]=0
    gaps=z[:,None]+foot[None,:,2]-heights
    valid=torch.isfinite(state).all(-1)&torch.isfinite(gaps).all(-1)&(points.abs()<terrain.HALF-2).all((1,2))
    valid&=(gaps.min(-1).values>=.0145)&(gaps.max(-1).values<=.20)
    # One device-to-host reduction per reset covers both guards. Same checks,
    # same thresholds; the failing guard is identified on the failure path only.
    if not bool(certified&valid.all()):
        if not bool(certified):raise RuntimeError('Soles outside certified slope-bound region')
        raise RuntimeError(f'Invalid local terrain spawn in worlds {ids[~valid][:16].cpu().tolist()}')
    robot.write_root_state_to_sim(state,env_ids=ids)
    q=robot.data.default_joint_pos[ids]
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
    robot.set_joint_position_target(q,env_ids=ids)
    if not hasattr(env,'_stand_target_root'):
        env._stand_target_root=torch.zeros(env.num_envs,3,device=env.device)
        env._stand_target_heading=torch.zeros(env.num_envs,4,device=env.device)
    env._stand_target_root[ids]=state[:,:3];env._stand_target_heading[ids]=state[:,3:7]
    env.scene.env_origins[ids,:2]=xy;env.scene.env_origins[ids,2]=ground.height(xy,ids)
    invalidate(env,ids)
    if not getattr(env,'_local_spawn_printed',False):
        print(f'[Local spawn verified] {len(ids)} worlds, native field selection; sole gaps '
              f'{float(gaps.min()):.4f}..{float(gaps.max()):.4f}m; finite patch boundary guarded',flush=True)
        env._local_spawn_printed=True


def patch_edge(env):
    return (env.scene['robot'].data.root_link_pos_w[:,:2].abs()>terrain.HALF-2).any(-1)


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    # Use the standard terrain entity for shared per-world origins; replace its
    # plane with native hfields in the scene callback before compilation.
    cfg.scene.env_spacing=0.
    cfg.scene.terrain=TerrainEntityCfg(terrain_type='plane',env_spacing=0.,textures=(),materials=())
    cfg.scene.spec_fn=configure_view if play else configure_scene
    cfg.events['reset_pose']=EventTermCfg(func=reset_local,mode='reset',params={'distinct_patches':play})
    cfg.terminations['terrain_edge']=TerminationTermCfg(func=patch_edge)
    # MJLab runs CPU mj_forward at the compiled default pose before reset_local.
    # put_data uploads these CPU contacts/constraints too, whose count is not
    # bounded by the GPU hfield collider's <=4 contacts/pair. The default pose
    # can intersect terrain (detail3 produced 27 CPU contacts). Keep the prior
    # terrain budget for both paths; tests cover the actual CPU upload state.
    # Runtime overflow_guard stays active. Sensor matching is GPU-only.
    cfg.sim.nconmax=128
    cfg.sim.njmax=512
    cfg.sim.contact_sensor_maxmatch=8
    return cfg


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT


class LunarRunner(StandRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.logger.log_dir:
            from terrain_provenance import save
            save(Path(self.logger.log_dir)/'terrain')

    def load(self,path,*args,**kwargs):
        source=Path(path).resolve()
        if 'wangyu_moon_stand_flat_v14' not in str(source):
            from terrain_provenance import verify
            verify(source,allow_transfer=os.environ.get('WANGYU_TERRAIN_TRANSFER')=='1')
        else:
            print(f'[Terrain transfer] flat v14 checkpoint -> {PHYSICS_REVISION}; current terrain, not exact replay',flush=True)
        return super().load(path,*args,**kwargs)


register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=LunarRunner)
