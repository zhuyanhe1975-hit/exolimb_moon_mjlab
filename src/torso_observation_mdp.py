"""v14 raw state: explicit native torso motion and standard projected gravity."""
import torch
from mjlab.envs.mdp.observations import projected_gravity
from mjlab.utils.lab_api.math import quat_apply_inverse
import no_margin_mdp as base
import torso_velocity

OBS_DIM=146
ROOT_GRAVITY=slice(46,49)
TORSO_ANGULAR_VELOCITY=slice(55,58)  # already in base: Torso frame, scale .5
TORSO_LINEAR_VELOCITY=slice(143,146)  # new: Torso COM, Torso frame, scale 1/.3


def local_torso_velocity(env):
    if not hasattr(env,'_torso_observation_quat_address'):
        sensor=env.sim.mj_model.sensor('robot/balance_torso_quat')
        if int(sensor.dim[0])!=4:raise ValueError('Expected native Torso frame quaternion dimension 4')
        env._torso_observation_quat_address=int(sensor.adr[0])
    start=env._torso_observation_quat_address
    quat=env.sim.data.sensordata[:,start:start+4]
    return quat_apply_inverse(quat,torso_velocity.velocity(env))


def observation(env):
    # Assemble v14 directly from the original native bundle: avoid allocating
    # an intermediate 143-column tensor only to slice and concatenate it again.
    state=base.base.observation(env)
    # Reuse the existing 3 columns rather than add a duplicate gravity vector.
    # G1 uses this standard MJLab term too: R_world_root^T * [0,0,-1].
    gravity=projected_gravity(env)
    linear=local_torso_velocity(env)/.3
    return torch.cat((state[:,:ROOT_GRAVITY.start],gravity,
                      state[:,ROOT_GRAVITY.stop:95],state[:,97:base.base.OBS_DIM],linear),-1)


def guarded_observation(env):
    from standing_diagnostics import guarded_observation as guard
    return guard(env,observation_fn=observation,obs_dim=OBS_DIM)
