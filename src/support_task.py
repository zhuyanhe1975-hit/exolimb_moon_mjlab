"""v10: recover projected support rather than enforce vertical body posture."""
from copy import deepcopy
import os
import math
import stand_pose_free_task as base
import support_mdp as mdp
import standing_diagnostics as diagnostics
from mjlab.sensor import ContactSensorCfg,ContactMatch
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.registry import register_mjlab_task
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Support-v10'
EXPERIMENT=os.environ.get('WANGYU_TRAIN_EXPERIMENT','wangyu_moon_stand_support_v10_contactsync1')
OBS_DIM=mdp.OBS_DIM
SPAWN_MARGIN,STATION_RADIUS=base.SPAWN_MARGIN,base.STATION_RADIUS
PUSH_PARAMS=base.PUSH_PARAMS
REWARD_WEIGHTS={**base.REWARD_WEIGHTS,'gravity_tilt':0.,'joint_pose':-.25,
    'pelvis_velocity':-.5,'centroidal_angular_momentum':-.002,
    'capture_risk':-.0,'airborne':-.2}


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    cfg.episode_length_s=30.
    cfg.scene.sensors=tuple(cfg.scene.sensors or ())+(
        ContactSensorCfg(name='support_contacts',
            primary=ContactMatch(mode='geom',pattern=('left_sole','right_sole'),entity='robot'),
            secondary=ContactMatch(mode='geom',pattern='site20_surface'),
            fields=('found','force','dist','pos'),reduce='maxforce',num_slots=4),)
    cfg.observations={group:ObservationGroupCfg({'state':ObservationTermCfg(func=diagnostics.guarded_observation)},enable_corruption=False)
                      for group in ('actor','critic')}
    for name,term in cfg.rewards.items():term.weight=REWARD_WEIGHTS[name]
    cfg.rewards['capture_risk']=RewardTermCfg(func=mdp.capture_risk,weight=REWARD_WEIGHTS['capture_risk'])
    cfg.rewards['airborne']=RewardTermCfg(func=mdp.airborne,weight=REWARD_WEIGHTS['airborne'])
    for name,column in (('contact_static_margin_m',0),('contact_capture_proxy_margin_m',1),
                        ('contact_slot_fraction',4),('support_has_area',5)):
        cfg.metrics[name]=MetricsTermCfg(func=mdp.metric,params={'column':column},reduce='mean')
    cfg.terminations={'numerical_guard':TerminationTermCfg(func=diagnostics.numerical_guard),**cfg.terminations}
    for name,func in (('joint_speed_rms_rad_s',diagnostics.joint_speed),
                      ('root_angular_speed_rad_s',diagnostics.root_angular_speed),
                      ('settled_unforced_fraction',diagnostics.settled_fraction),
                      ('moving_without_wrench_fraction',diagnostics.moving_without_wrench),
                      ('loaded_contact_slots',diagnostics.contact_count)):
        cfg.metrics[name]=MetricsTermCfg(func=func,reduce='mean')
    cfg.terminations['tilted'].params['limit_angle']=math.radians(60)
    cfg.terminations['torso_tilted'].func=mdp.torso_fallen
    return cfg


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=StandRunner)
