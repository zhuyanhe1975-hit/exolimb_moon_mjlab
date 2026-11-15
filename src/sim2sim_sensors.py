"""Native MuJoCo *kinematic* sensors driven exclusively by PhysX state.

No mj_step, mj_forward, inverse dynamics, contacts or force synthesis. Touch
measurements must come from the target physics engine. Native subtree velocity
and angular momentum include all original human bodies, excluding USD-only
auxiliary links. This avoids reimplementing articulated state estimation.
"""
from pathlib import Path
import mujoco
import numpy as np
from sim2sim_contract import height


class NativeSensors:
    def __init__(self, directory, manifest, skin=False):
        self.manifest=manifest
        self.model=mujoco.MjModel.from_xml_path(str(Path(directory)/('skin.xml' if skin else 'sensors.xml')))
        self.data=mujoco.MjData(self.model)
        self.grid=np.load(Path(directory)/'terrain.npz',allow_pickle=False)['grid']
        if not self.grid.size:self.grid=None
        self.half=manifest['terrain'].get('half_extent_m') or 8.
        self.target=np.array(manifest['root_spawn'][:3])
        self.heading=np.deg2rad(0.)
        q=np.array(manifest['root_spawn'][3:7]);self.heading=2*np.arctan2(q[3],q[0])
        self.sensors={name:self.model.sensor('balance_'+name) for name,*_ in manifest['sensor_layout']}
        self.torso_velocity=self.model.sensor('balance_torso_com_vel')
        actual=[self.model.joint(i).name for i in range(1,self.model.njnt)]
        if actual!=manifest['joint_names']:raise ValueError('Native sensor joint order differs from the policy')
        if not np.allclose(self.model.body_ipos[1],0):raise ValueError('Root COM must equal link origin in this contract')
        coordinates=np.linspace(-.5,.5,5)
        self.scan_points=np.array([(x,y) for x in coordinates for y in coordinates])

    def update(self, root_position, root_quaternion, world_linear_velocity, world_angular_velocity, q, qd):
        values=np.concatenate((root_position,root_quaternion,world_linear_velocity,world_angular_velocity,q,qd))
        if not np.isfinite(values).all():raise FloatingPointError('Nonfinite PhysX state; stopping inference')
        d=self.data;m=self.model
        d.qpos[:3]=root_position;d.qpos[3:7]=root_quaternion;d.qpos[7:]=q
        # MuJoCo freejoint linear velocity is world-frame, angular is body-frame.
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,np.asarray(root_quaternion,dtype=float))
        d.qvel[:3]=world_linear_velocity;d.qvel[3:6]=rotation.reshape(3,3).T@world_angular_velocity;d.qvel[6:]=qd
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d)
        mujoco.mj_subtreeVel(m,d);mujoco.mj_sensorPos(m,d);mujoco.mj_sensorVel(m,d)

    def value(self,name):
        sensor=self.sensors[name]
        return self.data.sensordata[int(sensor.adr[0]):int(sensor.adr[0])+int(sensor.dim[0])].copy()

    def raw(self, action, loads):
        m,d=self.model,self.data
        native=[]
        for name,*_ in self.manifest['sensor_layout']:
            native.extend([loads[0 if name=='left_load' else 1]] if name.endswith('_load') else self.value(name))
        root=self.value('root_pos');feet=[self.value(side+'_pos') for side in ('left','right')]
        q=self.value('root_quat');w,x,y,z=q
        c=1-2*(y*y+z*z);s=2*(w*z+x*y);norm=max(np.hypot(c,s),1e-6);c/=norm;s/=norm
        rotation=np.array([[c,-s],[s,c]])
        xy=self.scan_points@rotation.T+root[:2]
        scan=np.clip(5*(height(xy,self.grid,self.half)-height(root[None,:2],self.grid,self.half)[0]),-5,5)
        delta=root[:2]-self.target[:2];c0,s0=np.cos(self.heading),np.sin(self.heading)
        drift=np.array([c0*delta[0]+s0*delta[1],-s0*delta[0]+c0*delta[1]])
        ground=height(np.array([root[:2],feet[0][:2],feet[1][:2]]),self.grid,self.half)
        # The previous-action field feeds only unused diagnostics; actor sees the
        # last action applied to the just-completed control interval, once.
        raw=np.concatenate((d.qpos[7:]-np.array(self.manifest['q_default']),d.qvel[6:],action,action,
                            native,drift,ground,scan)).astype(np.float32)
        start=int(self.torso_velocity.adr[0])
        return raw[None],d.sensordata[start:start+3].astype(np.float32)[None]

    def touch_load(self,side,forces,points):
        """Normal contact loads inside the native TOUCH box (target engine only)."""
        sid=self.model.site('balance_'+side+'_sole').id
        local=(points-self.data.site_xpos[sid])@self.data.site_xmat[sid].reshape(3,3)
        inside=(np.abs(local)<=self.model.site_size[sid]+1e-6).all(-1)
        return float(np.abs(np.asarray(forces).reshape(-1)[inside]).sum())
