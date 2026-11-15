"""v9: v8 standing with zero initial-joint-angle penalty."""
from copy import deepcopy
import os
from mjlab.tasks.registry import register_mjlab_task
import stand_push_task as base
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Push-v9'
EXPERIMENT=os.environ.get('WANGYU_TRAIN_EXPERIMENT','wangyu_moon_stand_push_v9')
OBS_DIM=base.OBS_DIM
SPAWN_MARGIN,STATION_RADIUS=base.SPAWN_MARGIN,base.STATION_RADIUS
PUSH_PARAMS={**base.PUSH_PARAMS,
    'force_range':(-float(os.environ.get('WANGYU_TRAIN_PUSH_FORCE','30')),float(os.environ.get('WANGYU_TRAIN_PUSH_FORCE','30'))),
    'torque_range':(-float(os.environ.get('WANGYU_TRAIN_PUSH_TORQUE','0')),float(os.environ.get('WANGYU_TRAIN_PUSH_TORQUE','0')))}
REWARD_WEIGHTS={**base.REWARD_WEIGHTS,'joint_pose':-.25}


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    cfg.rewards['joint_pose'].weight=-0.25
    from push_options import apply
    apply(cfg,enabled=os.environ.get('WANGYU_TRAIN_PUSH_ENABLED')=='1',
          force=float(os.environ.get('WANGYU_TRAIN_PUSH_FORCE','30')),
          torque=float(os.environ.get('WANGYU_TRAIN_PUSH_TORQUE','0')))
    return cfg


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),
                    terrain_stand_rl,runner_cls=StandRunner)
