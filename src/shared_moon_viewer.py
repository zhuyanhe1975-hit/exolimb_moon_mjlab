"""One native display world: one DEM and independently skinned humans.

RL physics retains its independent local worlds. Only the CPU display model is
assembled here; native MuJoCo skinning uses distinct bones for each human. This
avoids mjv_addGeoms reusing one skin's vertex buffers for multiple environments.
"""
import numpy as np
import os
import mujoco
import torch
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
import local_lunar_terrain as terrain
import torso_velocity


class SharedMoonModel:
    def __init__(self,source,count,full_detail=True):
        self.count=count;self.bank=terrain.bank_data()
        spec=mujoco.MjSpec();spec.stat.extent=5.25
        spec.option.gravity=source.opt.gravity.copy()
        spec.worldbody.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[1,1,.1])
        terrain.configure_native(spec,full_view=True)
        lod_metadata=None
        if not full_detail:
            from lunar_view_lod import install
            lod_metadata=install(spec,self.bank)
        spec.delete(next(g for g in spec.geoms if g.name=='terrain'))
        # Move the one landscape from crop-0 coordinates back to map coordinates.
        spec.body('site20_ground').pos=np.r_[self.bank['centers'][0],self.bank['altitudes'][0]]
        template=torso_velocity.skin_spec()
        for index in range(count):
            robot=template.copy()
            for sensor in list(robot.sensors):robot.delete(sensor)
            for actuator in list(robot.actuators):robot.delete(actuator)
            for light in list(robot.lights):robot.delete(light)
            for geom in robot.geoms:geom.contype=geom.conaffinity=0
            for skin in robot.skins:
                skin.name=f'env_{index}/'+skin.name
                skin.material=f'env_{index}/'+skin.material.removeprefix('robot/')
            spec.attach(robot,prefix=f'env_{index}/',frame=spec.worldbody.add_frame())
        # Share texture/material assets; skins and bone bindings remain distinct.
        for skin in spec.skins:skin.material='env_0/'+skin.material.split('/',1)[1]
        for geom in spec.geoms:
            if geom.material.startswith('env_'):geom.material='env_0/'+geom.material.split('/',1)[1]
        for material in list(spec.materials):
            if material.name.startswith('env_') and not material.name.startswith('env_0/'):spec.delete(material)
        for texture in list(spec.textures):
            if texture.name.startswith('env_') and not texture.name.startswith('env_0/'):spec.delete(texture)
        spec.visual.map.znear=.002;spec.visual.map.zfar=1000.;spec.visual.map.shadowclip=1.5
        spec.visual.quality.shadowsize=4096
        # The passive UI processes its own A shortcut before the Python action
        # queue. Keep Auto Connect geometry invisible even during that one UI
        # frame; the viewer also actively clears mjVIS_AUTOCONNECT below.
        spec.visual.rgba.connect=[.2,.2,.8,0.]
        spec.visual.headlight.active=True
        spec.visual.headlight.ambient=[.12,.12,.12];spec.visual.headlight.diffuse=[.18,.18,.18]
        spec.body('env_0/Pelvis').add_light(name='shared_sun',type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK,pos=[-3,-4,6],dir=[.45,.55,-1],castshadow=True,
            diffuse=[.85,.82,.76],ambient=[.06,.06,.06],specular=[.25,.25,.25])
        self.model=spec.compile();self.data=mujoco.MjData(self.model)
        self.lod=None
        if lod_metadata is not None:
            from lunar_view_lod import DisplayLod
            self.lod=DisplayLod(self.model,lod_metadata)
        disable_model_sameframe_shortcuts(self.model)
        self.pelvis=np.array([self.model.body(f'env_{i}/Pelvis').id for i in range(count)])
        self.sun=self.model.light('shared_sun').id
        self.qpos_indices=[];self.qvel_indices=[]
        self.source_bodies=[];self.target_bodies=[];self.body_lookup={}
        for i in range(count):
            qi=[];vi=[];bodies=[]
            for joint in range(source.njnt):
                name=mujoco.mj_id2name(source,mujoco.mjtObj.mjOBJ_JOINT,joint)
                if not name or not name.startswith('robot/'):raise ValueError('Expected only named human joints')
                target=self.model.joint(f'env_{i}/'+name.removeprefix('robot/')).id
                kind=int(source.jnt_type[joint]);nq=7 if kind==0 else 4 if kind==1 else 1
                nv=6 if kind==0 else 3 if kind==1 else 1
                qi.extend(range(int(self.model.jnt_qposadr[target]),int(self.model.jnt_qposadr[target])+nq))
                vi.extend(range(int(self.model.jnt_dofadr[target]),int(self.model.jnt_dofadr[target])+nv))
            for body in range(source.nbody):
                name=mujoco.mj_id2name(source,mujoco.mjtObj.mjOBJ_BODY,body)
                if name and name.startswith('robot/'):
                    target=self.model.body(f'env_{i}/'+name.removeprefix('robot/')).id
                    bodies.append(target);self.body_lookup[target]=(i,body)
                    if i==0:self.source_bodies.append(body)
            self.qpos_indices.append(qi);self.qvel_indices.append(vi);self.target_bodies.append(bodies)
        self.qpos_indices=np.array(self.qpos_indices);self.qvel_indices=np.array(self.qvel_indices)
        self.target_bodies=np.array(self.target_bodies)
        if self.model.nq!=count*source.nq or self.model.nskin!=count*source.nskin:
            raise ValueError('Combined native human state/skin count mismatch')

    def update(self,qpos,qvel,wrenches,patches):
        """Translate native local root poses into original DEM coordinates."""
        positions=np.array(qpos,dtype=np.float64,copy=True)
        positions[:,:2]+=self.bank['centers'][patches]
        positions[:,2]+=self.bank['altitudes'][patches]
        self.data.qpos[self.qpos_indices]=positions
        self.data.qvel[self.qvel_indices]=qvel
        self.data.xfrc_applied[:]=0
        self.data.xfrc_applied[self.target_bodies]=wrenches[:,self.source_bodies]
        if self.lod is not None:self.lod.select(patches)


class SharedMoonViewer(NativeMujocoViewer):
    def setup(self):
        output=getattr(self,'_startup_output',None)
        if output is not None:output.resume()
        print(f'[Display] building one shared Moon scene for {self.env.unwrapped.num_envs} humans...',flush=True)
        if output is not None:output.start()
        try:
            import mujoco.viewer
            sim=self.env.unwrapped.sim
            self._shared=SharedMoonModel(sim.mj_model,self.env.unwrapped.num_envs,
                                        full_detail=os.environ.get('WANGYU_MOON_FULL_DETAIL_VIEW')=='1')
            self.mjm,self.mjd=self._shared.model,self._shared.data
            if self.cfg.fovy is not None:self.mjm.vis.global_.fovy=self.cfg.fovy
            self._last_perturbed_env=None
            self._overview=False
            self._sync_env_state_to_mjdata(self.mjd,sim.data,self.env_idx)
            mujoco.mj_forward(self.mjm,self.mjd)
            self.vopt=mujoco.MjvOption();self.pert=mujoco.MjvPerturb()
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]=False
            self._term_names=[name for name,_ in self.env.unwrapped.reward_manager.get_active_iterable_terms(self.env_idx)]
            self._init_reward_plots(self._term_names)
            self.viewer=mujoco.viewer.launch_passive(self.mjm,self.mjd,key_callback=self._safe_key_callback,
                                                    show_left_ui=False,show_right_ui=False)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]=False
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT]=False
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=int(self.cfg.enable_shadows)
            self.mjm.vis.map.force=.01;self.mjm.vis.scale.forcewidth=.04
            self._setup_camera()
        finally:
            output=getattr(self,'_startup_output',None)
            if output is not None:output.resume()
        print(f'[Shared Moon view] one Site20 map, {self._shared.count} independently skinned humans; '
              'source-map XY/yaw; A=map overview, comma/period=track human, mouse wheel=zoom',flush=True)
        if self._shared.lod is not None:
            print(f'[Display LOD] human patches=.1m, inactive patch interiors=1m; '
                  f'{self._shared.lod.triangles():,}/{self._shared.lod.metadata["original_triangles"]:,} terrain triangles drawn; '
                  'full map/skins/shadows retained; --full-detail-view restores all fine patches',flush=True)

    def _setup_camera(self):
        self._overview=False
        camera=self.viewer.cam
        camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid=int(self._shared.pelvis[self.env_idx])
        camera.fixedcamid=-1
        camera.distance=self.cfg.distance;camera.azimuth=self.cfg.azimuth;camera.elevation=self.cfg.elevation
        camera.lookat[:]=self.mjd.xpos[camera.trackbodyid]
        self.viewer.user_scn.ngeom=0

    def _sync_env_state_to_mjdata(self,target_data,sim_data,env_idx):
        del target_data,env_idx
        patches=self.env.unwrapped._local_lunar_surface.patch_ids.cpu().numpy()
        self._shared.update(sim_data.qpos.cpu().numpy(),sim_data.qvel.cpu().numpy(),
                            sim_data.xfrc_applied.cpu().numpy(),patches)

    def _sync_model_fields(self,sim,env_idx):
        # Never overwrite the one global landscape with a selected world's
        # local collision/display offsets from the GPU simulation model.
        del sim
        self.mjm.light_bodyid[self._shared.sun]=self._shared.pelvis[env_idx]

    def _render_other_env_geoms(self,*args):
        pass  # All people already own distinct native skins in this CPU model.

    def _add_env_selection_marker(self,viewer):
        pass  # Tracking/selection is identified by the native environment HUD.

    def _update_debug_visualizers(self,viewer):
        # A also toggles MuJoCo's built-in Auto Connect before our callback.
        # It creates blue auto-body geometry even with an empty user_scn.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT]=False
        viewer.user_scn.ngeom=0
        from wrench_visualization import add_torque_geoms
        add_torque_geoms(viewer.user_scn,self.mjd)

    def _handle_custom_action(self,action,payload):
        if action==ViewerAction.TOGGLE_SHOW_ALL_ENVS:
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT]=False
            self._overview=not self._overview
            if self._overview:
                positions=self.mjd.xpos[self._shared.pelvis]
                self.viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.lookat[:]=positions.mean(0)
                self.viewer.cam.distance=max(10.,np.linalg.norm(np.ptp(positions,axis=0))*1.2)
                if self._shared.count==1:
                    source=self._shared.bank['source']
                    self.viewer.cam.lookat[:]=[0,0,source['zmin_m']+.5*source['height_range_m']]
                    self.viewer.cam.distance=2.5*source['half_extent_m']
                self.viewer.cam.elevation=-65
            else:self._setup_camera()
            return True
        handled=super()._handle_custom_action(action,payload)
        if action in (ViewerAction.PREV_ENV,ViewerAction.NEXT_ENV):
            self._overview=False;self._setup_camera()
        return handled

    def sync_viewer_to_env(self):
        if self.viewer is None:return
        pert=self.viewer.perturb
        selection=self._shared.body_lookup.get(pert.select) if pert.active else None
        sim_data=self.env.unwrapped.sim.data
        if self._last_perturbed_env is not None:
            sim_data.qfrc_applied[self._last_perturbed_env]=0
            self._last_perturbed_env=None
        if selection is None:return
        index,_=selection
        mujoco.mjv_applyPerturbForce(self.mjm,self.mjd,pert)
        force,torque=np.split(self.mjd.xfrc_applied[pert.select].copy(),2)
        generalized=np.zeros(self.mjm.nv)
        mujoco.mj_applyFT(self.mjm,self.mjd,force,torque,self.mjd.xipos[pert.select],pert.select,generalized)
        sim_data.qfrc_applied[index]=torch.as_tensor(generalized[self._shared.qvel_indices[index]],
                                                   device=sim_data.qfrc_applied.device,dtype=sim_data.qfrc_applied.dtype)
        self._last_perturbed_env=index
        self.mjd.xfrc_applied[pert.select]=0

    def close(self):
        previous=getattr(self,'_last_perturbed_env',None)
        if previous is not None:
            self.env.unwrapped.sim.data.qfrc_applied[previous]=0
            self._last_perturbed_env=None
        super().close()
