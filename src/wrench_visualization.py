"""Native scene decoration for actual world-frame applied torque; no physics writes."""
import numpy as np
import mujoco


def add_torque_geoms(scene, data):
    color=np.array([.85,.2,1.,1.],dtype=np.float32)

    def connector(kind,start,end,width,label=''):
        if scene.ngeom>=scene.maxgeom:
            return
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),color)
        mujoco.mjv_connector(geom,kind,width,start,end)
        geom.label=label
        scene.ngeom+=1

    for body in range(1,len(data.xfrc_applied)):
        torque=np.asarray(data.xfrc_applied[body,3:])
        magnitude=float(np.linalg.norm(torque))
        if not np.isfinite(magnitude) or magnitude<1e-6:
            continue
        axis=torque/magnitude
        center=data.xipos[body]
        connector(mujoco.mjtGeom.mjGEOM_ARROW,center,
                  center+axis*min(.7,.03*magnitude),.012,
                  f'T={magnitude:.1f} Nm')
        # Positive angle about the torque axis follows the right-hand rule.
        reference=np.eye(3)[np.argmin(np.abs(axis))]
        u=np.cross(axis,reference);u/=np.linalg.norm(u)
        v=np.cross(axis,u)
        angles=np.linspace(0,1.5*np.pi,19)
        points=center+.24*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)
        for index in range(len(points)-1):
            kind=mujoco.mjtGeom.mjGEOM_ARROW if index==len(points)-2 else mujoco.mjtGeom.mjGEOM_CAPSULE
            connector(kind,points[index],points[index+1],.014 if index==len(points)-2 else .004)
