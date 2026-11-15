"""Whole-human linear momentum opposite gravity, calibrated by ballistic rise."""
import math
import numpy as np
import torch



def anti_gravity_direction(gravity):
    gravity=np.asarray(gravity,dtype=np.float64)
    magnitude=float(np.linalg.norm(gravity))
    if gravity.shape!=(3,) or not np.isfinite(gravity).all() or magnitude<=1e-8:
        raise ValueError('Upward-speed reward requires a finite nonzero gravity vector')
    return -gravity/magnitude


def upward_speed(env):
    # SUBTREELINVEL(Pelvis) includes the full human, not just root or torso.
    # Its native output is WORLD m/s; model.opt.gravity is WORLD m/s^2.
    if not hasattr(env,'_jump_com_velocity_address'):
        sensor=env.sim.mj_model.sensor('robot/balance_com_vel')
        if int(sensor.dim[0])!=3:raise ValueError('Expected native subtree COM velocity dimension 3')
        env._jump_com_velocity_address=int(sensor.adr[0])
    address=env._jump_com_velocity_address
    velocity=env.sim.data.sensordata[:,address:address+3]
    # This MJLab version initializes EntityData.gravity_vec_w to fixed -Z;
    # it cannot express arbitrary native gravity directions. Use model.opt.gravity.
    gravity=tuple(float(x) for x in env.sim.mj_model.opt.gravity)
    if getattr(env,'_jump_gravity_key',None)!=gravity:
        env._jump_up_direction=torch.as_tensor(anti_gravity_direction(gravity),
                                               device=velocity.device,dtype=velocity.dtype)
        env._jump_gravity_key=gravity
    direction=env._jump_up_direction
    return (velocity*direction).sum(-1)


def subtree_mass(env):
    if not hasattr(env,'_jump_pelvis_body_id'):
        env._jump_pelvis_body_id=env.sim.mj_model.body('robot/Pelvis').id
        mass=float(env.sim.mj_model.body_subtreemass[env._jump_pelvis_body_id])
        if not np.isfinite(mass) or mass<=0:raise ValueError('Pelvis subtree mass must be finite and positive')
    root=env._jump_pelvis_body_id
    # MuJoCo has no SUBTREELINMOM sensor. Use its native total subtree mass
    # times native SUBTREELINVEL; never sum or weight individual body states.
    # Read the backend field when present, including per-world mass variants.
    model=getattr(env.sim,'model',None)
    if model is not None and hasattr(model,'body_subtreemass'):
        return model.body_subtreemass[...,root]
    return float(env.sim.mj_model.body_subtreemass[root])


def target_upward_speed(env,jump_height):
    from jump_settings import validate_height
    height=validate_height(jump_height)
    gravity=float(np.linalg.norm(env.sim.mj_model.opt.gravity))
    anti_gravity_direction(env.sim.mj_model.opt.gravity) # validate native gravity
    return float(np.sqrt(2*gravity*height))


def upward_momentum(env):
    """Signed world-gravity-opposite linear momentum, kg*m/s."""
    return subtree_mass(env)*upward_speed(env)


def target_momentum(env,jump_height):
    """Ideal takeoff momentum for a COM ballistic rise h; kg*m/s."""
    speed=upward_speed(env)
    mass=torch.as_tensor(subtree_mass(env),device=speed.device,dtype=speed.dtype)
    return (mass*target_upward_speed(env,jump_height)).expand_as(speed)


def exponential_momentum_fraction(ratio,exponent):
    """Convex reward: idle=0, target=1; low momenta earn much less than linear.

    A bounded exponent keeps expm1 finite in float32. It changes reward shape,
    not physics momentum or the ordinary RL accumulation of per-step rewards.
    """
    if not math.isfinite(exponent) or not 0<exponent<=50:
        raise ValueError('momentum exponent must be finite and in (0, 50]')
    x=ratio.clamp(0.,1.)
    return (torch.expm1(exponent*x)/math.expm1(exponent)).clamp(0.,1.)


def momentum_reward(env,jump_height,exponent=5.):
    """Dense instantaneous exponential ratio; no contact/history gating."""
    ratio=upward_momentum(env)/target_momentum(env,jump_height)
    return exponential_momentum_fraction(ratio,exponent)


def height_data(env):
    """Native COM rise from the episode's first supported snapshot.

    Baseline bookkeeping only: never reconstruct a COM, integrate velocity, or
    advance physics. This is COM displacement, not a foot/terrain clearance.
    Reward/metrics share one pre-reset sample; observations never update it.
    """
    key=(env.common_step_counter,getattr(env,'_stand_reset_generation',0))
    if getattr(env,'_jump_height_key',None)==key:return env._jump_height_values
    if not hasattr(env,'_jump_com_position_address'):
        sensor=env.sim.mj_model.sensor('robot/balance_com_pos')
        if int(sensor.dim[0])!=3:raise ValueError('Expected native whole-human subtree COM position')
        env._jump_com_position_address=int(sensor.adr[0])
    address=env._jump_com_position_address
    com=env.sim.data.sensordata[:,address:address+3]
    from support_mdp import phase_contact_data
    contact=phase_contact_data(env,'reward')
    touching=(contact.found>0)&(contact.dist<=.002)&torch.isfinite(contact.pos).all(-1)
    supported=touching.any(-1)
    done=getattr(env,'reset_buf',torch.zeros_like(supported)).bool()
    if not hasattr(env,'_jump_height_baseline'):
        env._jump_height_baseline=torch.zeros_like(com)
        env._jump_height_ready=torch.zeros_like(supported)
    first=supported&~env._jump_height_ready&~done
    env._jump_height_baseline.copy_(torch.where(first[:,None],com,env._jump_height_baseline))
    env._jump_height_ready|=first
    # Direction from actual native gravity; keep the same world frame as COM.
    direction=torch.as_tensor(anti_gravity_direction(env.sim.mj_model.opt.gravity),device=com.device,dtype=com.dtype)
    rise=((com-env._jump_height_baseline)*direction).sum(-1).clamp_min(0.)
    ready=env._jump_height_ready
    rise=torch.where(ready,rise,torch.zeros_like(rise))
    airborne=ready&~supported&~done
    result={'rise':rise,'airborne':airborne,'ready':ready.clone()}
    env._jump_height_key=key;env._jump_height_values=result
    return result


def height_reward(env,jump_height,exponent=5.):
    from jump_settings import validate_height
    sample=height_data(env)
    return sample['airborne'].float()*exponential_momentum_fraction(sample['rise']/validate_height(jump_height),exponent)


def measured_com_rise(env):return height_data(env)['rise']
def height_airborne_fraction(env):return height_data(env)['airborne'].float()
def measured_airborne_com_rise(env):
    sample=height_data(env)
    return sample['rise']*sample['airborne']
