"""v14: no-margin plane task with direct Torso COM velocity observation."""
from copy import deepcopy
import os
import flat_no_margin_task as base
import torso_observation_mdp as mdp
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Flat-v14'
EXPERIMENT=os.environ.get('WANGYU_TRAIN_EXPERIMENT','wangyu_moon_stand_flat_v14')
OUTPUT_SUBDIR='stand_flat_v14'
IS_FLAT=True
SUPPORT_PROXY_ENABLED=False
OBSERVATION_DESCRIPTION='root projected gravity:3 (MJLab/G1); Torso angular:3; Torso COM linear:3; local frames'
OBS_DIM=mdp.OBS_DIM
SPAWN_MARGIN,STATION_RADIUS=base.SPAWN_MARGIN,base.STATION_RADIUS
PUSH_PARAMS=base.PUSH_PARAMS
CLEARANCE=base.CLEARANCE
REWARD_WEIGHTS=dict(base.REWARD_WEIGHTS)


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    cfg.observations={group:ObservationGroupCfg({'state':ObservationTermCfg(func=mdp.guarded_observation)},
        enable_corruption=False) for group in ('actor','critic')}
    return cfg


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=StandRunner)
