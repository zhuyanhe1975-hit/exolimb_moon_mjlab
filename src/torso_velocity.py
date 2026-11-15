"""Native world-frame velocity of the Torso rigid body's inertial COM."""
import mujoco
import balance_task as physical

SENSOR_NAME='balance_torso_com_vel'


def add_sensor(spec):
    # BODY is the inertial frame at the rigid-body COM. XBODY / a site at
    # Torso origin would measure near the waist pivot and can miss rotation.
    spec.add_sensor(name=SENSOR_NAME,type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,objname='Torso')
    return spec


def robot_spec():
    return add_sensor(physical.robot_spec())


def skin_spec():
    return add_sensor(physical.skin_spec())


def velocity(env):
    if not hasattr(env,'_torso_velocity_address'):
        sensor=env.sim.mj_model.sensor('robot/'+SENSOR_NAME)
        if int(sensor.dim[0])!=3:raise ValueError('Expected native Torso COM velocity dimension 3')
        env._torso_velocity_address=int(sensor.adr[0])
    start=env._torso_velocity_address
    return env.sim.data.sensordata[:,start:start+3]


def squared(env):
    return velocity(env).square().sum(-1)


def speed(env):
    return velocity(env).norm(dim=-1)


def angular_velocity(env):
    # Existing native FRAMEANGVEL at balance_torso: angular velocity is
    # independent of whether measured at the waist origin or inertial COM.
    if not hasattr(env,'_torso_angular_velocity_address'):
        sensor=env.sim.mj_model.sensor('robot/balance_torso_omega')
        if int(sensor.dim[0])!=3:raise ValueError('Expected native Torso angular velocity dimension 3')
        env._torso_angular_velocity_address=int(sensor.adr[0])
    start=env._torso_angular_velocity_address
    return env.sim.data.sensordata[:,start:start+3]


def angular_squared(env):
    return angular_velocity(env).square().sum(-1)


def angular_speed(env):
    return angular_velocity(env).norm(dim=-1)
