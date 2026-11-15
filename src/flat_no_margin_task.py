"""v13: native-plane standing, no margin reward or margin/prediction inputs."""
from copy import deepcopy
import os
import flat_support_task as base
import no_margin_mdp as mdp
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Flat-v13'
EXPERIMENT=os.environ.get('WANGYU_TRAIN_EXPERIMENT','wangyu_moon_stand_flat_v13')
OUTPUT_SUBDIR='stand_flat_v13'
IS_FLAT=True
SUPPORT_PROXY_ENABLED=False
OBS_DIM=mdp.OBS_DIM
SPAWN_MARGIN=base.SPAWN_MARGIN
STATION_RADIUS=base.STATION_RADIUS
PUSH_PARAMS=base.PUSH_PARAMS
CLEARANCE=base.CLEARANCE
REWARD_WEIGHTS={name:weight for name,weight in base.REWARD_WEIGHTS.items() if name!='capture_risk'}


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    cfg.rewards.pop('capture_risk',None)
    cfg.rewards['airborne'].func=mdp.airborne
    cfg.observations={group:ObservationGroupCfg({'state':ObservationTermCfg(
        func=mdp.guarded_observation)},
        enable_corruption=False) for group in ('actor','critic')}
    for name in ('contact_static_margin_m','contact_capture_proxy_margin_m','contact_slot_fraction',
                 'support_has_area','predicted_support_offset_m'):
        cfg.metrics.pop(name,None)
    return cfg


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=StandRunner)
