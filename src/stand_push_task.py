"""Native random pelvis force pulses for non-LQR lunar standing (v8).

Reuse MJLab apply_body_impulse scheduling, duration, reset and wrench API.
The installed standard term samples all XYZ components with the same scalar
range; this small adapter supplies tensor bounds to its native sampler so Fz=0.
"""
from copy import deepcopy
import os

import torch
from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task

import balance_task as base
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Push-v8'
EXPERIMENT=os.environ.get('WANGYU_TRAIN_EXPERIMENT','wangyu_moon_stand_push_v8')
OBS_DIM=base.OBS_DIM
SPAWN_MARGIN,STATION_RADIUS=base.SPAWN_MARGIN,base.STATION_RADIUS
REWARD_WEIGHTS=dict(base.REWARD_WEIGHTS)
PUSH_PARAMS={'force_range':(-30.,30.),'torque_range':(0.,0.),
             'duration_s':(.10,.20),'cooldown_s':(2.,4.)}


class HorizontalBodyImpulse(apply_body_impulse):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        low,high=cfg.params['force_range']
        self._horizontal_bounds=(
            torch.tensor([low,low,0.],device=self._device,dtype=torch.float32),
            torch.tensor([high,high,0.],device=self._device,dtype=torch.float32))
        print(f'[Pelvis push] Fx/Fy={low:g}..{high:g}N, Fz=0; '
              f'Tx/Ty/Tz={cfg.params["torque_range"]}Nm; '
              f'duration={cfg.params["duration_s"]}s, cooldown={cfg.params["cooldown_s"]}s; '
              'independent timers, cleared on reset; no LQR',flush=True)

    def __call__(self,env,env_ids,force_range,torque_range,duration_s,cooldown_s,
                 asset_cfg,body_point_offset=None):
        del force_range  # Bounds captured at init, like the native cooldown.
        return super().__call__(env,env_ids,force_range=self._horizontal_bounds,
            torque_range=torque_range,duration_s=duration_s,cooldown_s=cooldown_s,
            asset_cfg=asset_cfg,body_point_offset=body_point_offset)


def push_force(env,asset_cfg):
    # Native world-frame xfrc_applied view, not policy actions or inferred force.
    robot=env.scene[asset_cfg.name]
    return robot.data.body_external_wrench[:,asset_cfg.body_ids,:3].norm(dim=-1).sum(-1)


def push_active(env,asset_cfg):
    return (push_force(env,asset_cfg)>1e-6).float()


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    from push_options import apply
    apply(cfg,enabled=os.environ.get('WANGYU_TRAIN_PUSH_ENABLED')=='1',
          force=float(os.environ.get('WANGYU_TRAIN_PUSH_FORCE','30')),
          torque=float(os.environ.get('WANGYU_TRAIN_PUSH_TORQUE','0')))
    for name,func in (('pelvis_push_force_N',push_force),('pelvis_push_active_fraction',push_active)):
        cfg.metrics[name]=MetricsTermCfg(func=func,
            params={'asset_cfg':SceneEntityCfg('robot',body_names=('Pelvis',))},reduce='mean')
    return cfg


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),
                    terrain_stand_rl,runner_cls=StandRunner)
