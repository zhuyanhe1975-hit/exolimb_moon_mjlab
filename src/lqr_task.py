"""Opt-in v5: local flat-ground LQR + terrain-adapting residual PPO.

Registration is static: it never prepares a controller or steps physics.
Default v4 and its checkpoints are unchanged.
"""
from copy import deepcopy

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task

import balance_task as base
from lqr_action import ResidualLqrActionCfg, observation, final_action_change, metric
from lqr_artifact import STATE_DIM, NU
from lqr_runner import LqrRunner

TASK_ID = 'Wangyu-Moon-Stand-LQR-v5'
EXPERIMENT = 'wangyu_moon_stand_lqr_v5'
OBS_DIM = base.OBS_DIM + STATE_DIM + 2*NU + 2
SPAWN_MARGIN, STATION_RADIUS = base.SPAWN_MARGIN, base.STATION_RADIUS
REWARD_WEIGHTS = dict(base.REWARD_WEIGHTS)


def terrain_stand_cfg(play=False):
    cfg = base.terrain_stand_cfg(play=play)
    clip = cfg.actions['position'].clip
    cfg.actions = {'position':ResidualLqrActionCfg(entity_name='robot',
        actuator_names=('.*',), scale=1., clip=clip, residual_scale=.05)}
    cfg.observations = {name:ObservationGroupCfg(
        {'state':ObservationTermCfg(func=observation)},enable_corruption=False)
        for name in ('actor','critic')}
    cfg.rewards['action_change'] = RewardTermCfg(func=final_action_change,
                                                weight=REWARD_WEIGHTS['action_change'])
    for name, key in (('lqr_authority','authority'),('target_clipped_env_fraction','saturation'),
                      ('rl_residual_rms_rad','residual_rms')):
        cfg.metrics[name] = MetricsTermCfg(func=metric,params={'name':key},reduce='mean')
    return cfg


terrain_stand_rl = deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name = EXPERIMENT
# Small initial exploration around an explicitly zero-initialized residual mean.
terrain_stand_rl.actor.distribution_cfg['init_std'] = .15
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),
                    terrain_stand_rl,runner_cls=LqrRunner)
