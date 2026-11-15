"""Minimal independent Wangyu standing experiment, 20 axes, flat lunar ground."""
from copy import deepcopy
import math
import mujoco
import torch

from moon_task import native_cfg, native_spec, native_skin_spec, native_rl
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import time_out, bad_orientation, root_height_below_minimum
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from stand_compute import bundle, invalidate, position_error
from stand_runner import StandRunner
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task

TASK_ID = 'Wangyu-Moon-Stand-v1'
EXPERIMENT = 'wangyu_moon_stand_v1'
GRAVITY = 9.81 / 6
ACTION_SCALE = .10
ACTION_LIMIT = .15


def _position_spec(s):
    """Implicit native joint PD; preserve human effort limits and all geometry."""
    s.add_exclude(name='no_foot_self_collision', bodyname1='LeftFoot', bodyname2='RightFoot')
    for actuator in s.actuators:
        lower = any(word in actuator.name for word in ('Hip', 'Knee', 'Ankle'))
        limit = 180. if lower else 60.
        actuator.set_to_position(kp=300. if lower else 100., kv=25. if lower else 10.)
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange = [-limit, limit]
    return s


def stand_spec():
    return _position_spec(native_spec())


def stand_skin_spec():
    return _position_spec(native_skin_spec())


def pose_error(env):
    return position_error(env)


def observation(env):
    return bundle(env)[:, :72]


def stability(env):
    return bundle(env)[:, 72]


def action_change(env):
    return (env.action_manager.action - env.action_manager.prev_action).square().mean(-1)


def gravity_tilt_deg(env):
    """Inclination to gravity in degrees; yaw-invariant, diagnostics only."""
    # Computed once with observations/reward inside the existing tensor graph;
    # both mean and peak metrics consume the same owned output.
    return bundle(env)[:, 73]


def stand_cfg(play=False):
    c = native_cfg(play=play)
    robot = c.scene.entities['robot']
    robot.spec_fn = stand_skin_spec if play else stand_spec
    robot.init_state.joint_pos = {'.*': 0.0}
    robot.init_state.joint_vel = {'.*': 0.0}
    # Clip position targets in radians AFTER scaling; keep within hinge limits.
    physical = native_spec().compile()
    clip = {physical.joint(i).name: (max(-ACTION_LIMIT, float(physical.jnt_range[i, 0])),
                                   min(ACTION_LIMIT, float(physical.jnt_range[i, 1])))
            for i in range(1, physical.njnt)}
    c.actions = {'position': JointPositionActionCfg(entity_name='robot', actuator_names=('.*',),
                                                   scale=ACTION_SCALE, clip=clip)}
    c.observations = {group: ObservationGroupCfg({'state': ObservationTermCfg(func=observation)},
                                                enable_corruption=False)
                      for group in ('actor', 'critic')}
    c.rewards = {'stability': RewardTermCfg(func=stability, weight=1.),
                 'action_change': RewardTermCfg(func=action_change, weight=-.002)}
    c.metrics = {
        'gravity_tilt_mean_deg': MetricsTermCfg(func=gravity_tilt_deg, reduce='mean'),
        'gravity_tilt_peak_deg': MetricsTermCfg(func=gravity_tilt_deg, reduce='max'),
    }
    initial_height = robot.init_state.pos[2]
    c.terminations = {
        'time_out': TerminationTermCfg(func=time_out, time_out=True),
        'tilted': TerminationTermCfg(func=bad_orientation, params={'limit_angle': math.radians(30)}),
        'fallen': TerminationTermCfg(func=root_height_below_minimum,
                                    params={'minimum_height': initial_height * .70}),
    }
    c.scene.num_envs = 1 if play else 8192
    c.scene.env_spacing = 3.
    c.episode_length_s = 1e6 if play else 10.
    c.sim.mujoco.gravity = (0., 0., -GRAVITY)
    # Two box soles vs plane: at most eight contacts; retain headroom for limits.
    c.sim.nconmax = 16
    c.sim.njmax = 96
    c.sim.broadphase = 'nxn'
    c.events = {'reset_pose': EventTermCfg(func=reset_scene_to_default, mode='reset'),
                'invalidate_bundle': EventTermCfg(func=invalidate, mode='reset')}
    # No pushes or reset randomization. Explicit reset places each world at its origin.
    c.viewer.distance = 3.5
    c.viewer.azimuth = 135.
    c.viewer.elevation = -12.
    return c


stand_rl = deepcopy(native_rl)
stand_rl.experiment_name = EXPERIMENT
stand_rl.max_iterations = 200
stand_rl.save_interval = 25
stand_rl.num_steps_per_env = 32
stand_rl.actor.distribution_cfg['init_std'] = .3
stand_rl.algorithm.entropy_coef = .001
stand_rl.algorithm.learning_rate = .0003
stand_rl.clip_actions = 1.5
stand_rl.upload_model = False
register_mjlab_task(TASK_ID, stand_cfg(), stand_cfg(play=True), stand_rl, runner_cls=StandRunner)
