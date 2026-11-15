"""Thin Kit adapter around the existing shared FollowCamera implementation."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from collections import deque
import numpy as np


class Sim2SimCamera:
    def __init__(self,stage,positions,dt,enabled=True):
        from pxr import UsdGeom,Gf
        import carb.input
        import omni.appwindow
        from omni.kit.viewport.utility import get_active_viewport
        reference=Path(os.environ.get('ZBOT_MOON_DIR',str(Path.home()/'myWorks/zbot_exoarm_moon')))/'experiments/apollo_lunar/isaaclab_task/tools/follow_camera.py'
        if not reference.is_file():raise FileNotFoundError(f'Shared follow camera missing: {reference}')
        spec=importlib.util.spec_from_file_location('wangyu_shared_follow_camera',reference)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.stage=stage;self.dt=dt;self.requests=deque();self.overview=False
        self.saved_view=None;self.binding=None
        camera=UsdGeom.Camera.Define(stage,'/World/Camera')
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,20000.))
        viewport=get_active_viewport()
        if viewport:viewport.camera_path='/World/Camera'
        sim=SimpleNamespace(visualizers=[SimpleNamespace(cfg=SimpleNamespace(visualizer_type='kit'))],
                            set_camera_view=self.set_view)
        env=SimpleNamespace(sim=sim,num_envs=len(positions),step_dt=dt)
        self.follow=module.FollowCamera(env,initialize_view=False,enabled=enabled)
        first=np.array(positions[0]);self.set_view(first+np.array([3,-4,1.4]),first+np.array([0,0,.1]))
        window=omni.appwindow.get_default_app_window()
        if window is not None:
            interface=carb.input.acquire_input_interface();keyboard=window.get_keyboard()
            def callback(event,*_):
                if event.type==carb.input.KeyboardEventType.KEY_PRESS and event.input.name=='A':
                    self.requests.append('overview');return False
                return True
            subscription=interface.subscribe_to_keyboard_events(keyboard,callback)
            self.binding=(interface,keyboard,subscription,callback)
        print('[Camera] A=overview/near view; backslash=follow; comma/period=previous/next human; mouse orbit/zoom',flush=True)
        if len(positions)>1 and not enabled:
            self.requests.append('overview');self.update(positions)

    def set_view(self,eye,target):
        from pxr import UsdGeom,Gf,Sdf
        xform=UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/Camera'))
        destination=Gf.Vec3d(*map(float,target))
        matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),destination,Gf.Vec3d(0,0,1)).GetInverse()
        xform.MakeMatrixXform().Set(matrix)
        xform.GetPrim().CreateAttribute('omni:kit:centerOfInterest',Sdf.ValueTypeNames.Double3).Set(matrix.GetInverse().Transform(destination))

    def update(self,positions,dt=None):
        self.follow.poll_controls()
        while self.requests:
            self.requests.popleft();self.overview=not self.overview
            if self.overview:
                self.saved_view=self.follow.io.view();self.follow.enabled=False
                lo=np.min(positions,axis=0);hi=np.max(positions,axis=0);center=(lo+hi)/2
                span=max(10.,float(np.linalg.norm(hi-lo))*1.2)
                self.set_view(center+np.array([0,-.4*span,span]),center)
            else:
                position=np.array(positions[self.follow.env_index]);self.follow.enabled=True
                if self.saved_view is None:eye=np.array([3,-4,1.4]);target=np.array([0,0,.1])
                else:eye=np.array(self.saved_view[0])-np.array(self.saved_view[1]);target=np.array([0,0,.1])
                self.set_view(position+target+eye,position+target)
                self.follow.filter.position=tuple(position)
        self.follow.update(position=positions[self.follow.env_index],dt=self.dt if dt is None else dt)

    def close(self):
        self.follow.close()
        if self.binding is not None:
            interface,keyboard,subscription,_=self.binding
            interface.unsubscribe_to_keyboard_events(keyboard,subscription);self.binding=None
