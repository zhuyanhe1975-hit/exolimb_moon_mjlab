"""Restore Apollo waist yaw and ankle roll on the preserved source-pose asset.
Legacy 20-axis files remain untouched. Head and wrists remain fixed.
"""
import math
import mujoco


def restore_23_axes(spec):
    for body,name,axis,limit in (
        ('Torso','WaistYaw',[0,0,1],40.),
        ('LeftFoot','LeftAnkleRoll',[1,0,0],30.),
        ('RightFoot','RightAnkleRoll',[1,0,0],30.),
    ):
        spec.body(body).add_joint(name=name,type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=[0,0,0],axis=axis,ref=0.,limited=True,
            range=[-math.radians(limit),math.radians(limit)],damping=2.,armature=.05)
        lower='Ankle' in name
        actuator=spec.add_actuator(name=name,target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,gear=[1,0,0,0,0,0])
        actuator.set_to_position(kp=300. if lower else 100.,kv=25. if lower else 10.)
        actuator.ctrllimited=False
        actuator.forcelimited=True
        actuator.forcerange=[-180.,180.] if lower else [-60.,60.]
    return spec
