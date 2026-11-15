"""Native CPU display-world/skin/force mapping regression; no simulation steps."""
from types import SimpleNamespace
from contextlib import nullcontext
import unittest
import warnings
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.viewer.base import ViewerAction
import lunar_finetune_task as task
import local_lunar_terrain as terrain
from shared_moon_viewer import SharedMoonModel,SharedMoonViewer


class SharedMoonView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.source=Scene(cfg.scene,'cpu').compile()
            cls.shared=SharedMoonModel(cls.source,2)

    def setUp(self):
        self.patches=np.array([0,3])
        self.qpos=np.tile(self.source.qpos0,(2,1))
        self.qpos[:,0]=[.2,-.3];self.qpos[:,1]=[-.4,.1]
        self.qvel=np.zeros((2,self.source.nv))
        self.wrenches=np.zeros((2,self.source.nbody,6))
        self.shared.update(self.qpos,self.qvel,self.wrenches,self.patches)
        mujoco.mj_forward(self.shared.model,self.shared.data)

    def test_one_map_and_separate_native_skin_bones(self):
        shared=self.shared;model=shared.model
        self.assertEqual(model.ntex,self.source.ntex)
        self.assertEqual(model.nhfield,self.source.nhfield)
        self.assertEqual(model.nskin,2*self.source.nskin)
        self.assertEqual(sum(g.name=='site20_full_visual' for g in (model.geom(i) for i in range(model.ngeom))),1)
        for skin in range(model.nskin):
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_SKIN,skin)
            prefix=name.split('/')[0]+'/'
            start=int(model.skin_boneadr[skin]);count=int(model.skin_bonenum[skin])
            for body in model.skin_bonebodyid[start:start+count]:
                self.assertTrue(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(body)).startswith(prefix))
        for i,patch in enumerate(self.patches):
            data=mujoco.MjData(self.source);data.qpos[:]=self.qpos[i]
            mujoco.mj_kinematics(self.source,data)
            offset=np.r_[shared.bank['centers'][patch],shared.bank['altitudes'][patch]]
            np.testing.assert_allclose(shared.data.xpos[shared.target_bodies[i]],
                                       data.xpos[shared.source_bodies]+offset,atol=1e-10,rtol=0)

    def test_global_map_ray_matches_local_collision_height(self):
        shared=self.shared;model=shared.model
        ground=terrain.LocalSurface(SimpleNamespace(sim=SimpleNamespace(mj_model=self.source),num_envs=2,device='cpu'))
        ground.patch_ids[:]=torch.tensor(self.patches)
        model.geom_group[:]=0
        for i in range(model.ngeom):
            if model.geom(i).name.startswith('site20_'):model.geom_group[i]=5
        for i,patch in enumerate(self.patches):
            xy=np.array([.553,-.431]);center=shared.bank['centers'][patch];alt=shared.bank['altitudes'][patch]
            distance=mujoco.mj_ray(model,shared.data,np.r_[center+xy,alt+10.],np.array([0.,0.,-1.]),
                np.array([0,0,0,0,0,1],dtype=np.uint8),True,-1,None)
            expected=float(ground.height(torch.tensor(xy[None,:],dtype=torch.float32),torch.tensor([i]))[0])
            self.assertAlmostEqual(10-distance,expected,delta=5e-5)

    def test_mouse_wrench_maps_to_only_its_human(self):
        shared=self.shared;data=mujoco.MjData(self.source);data.qpos[:]=self.qpos[1]
        mujoco.mj_forward(self.source,data)
        source_body=self.source.body('robot/Torso').id
        target=shared.model.body('env_1/Torso').id
        force=np.array([12.,-5.,3.]);torque=np.array([2.,3.,-1.])
        original=np.zeros(self.source.nv);combined=np.zeros(shared.model.nv)
        mujoco.mj_applyFT(self.source,data,force,torque,data.xipos[source_body],source_body,original)
        mujoco.mj_applyFT(shared.model,shared.data,force,torque,shared.data.xipos[target],target,combined)
        np.testing.assert_allclose(combined[shared.qvel_indices[1]],original,atol=1e-10,rtol=0)
        np.testing.assert_array_equal(combined[shared.qvel_indices[0]],0)

    def test_overview_zoom_and_follow_never_leave_solid_markers(self):
        view=SharedMoonViewer.__new__(SharedMoonViewer)
        view._shared=self.shared;view.mjd=self.shared.data;view.env_idx=0
        view.cfg=SimpleNamespace(distance=3.5,azimuth=135,elevation=-22)
        view.viewer=SimpleNamespace(cam=mujoco.MjvCamera(),user_scn=mujoco.MjvScene(self.shared.model,maxgeom=64),
                                    opt=mujoco.MjvOption(),lock=nullcontext)
        view.viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        view.viewer.cam.distance=2500;view._overview=True
        view._update_debug_visualizers(view.viewer)
        scene=view.viewer.user_scn
        self.assertEqual(scene.ngeom,0)
        # The passive viewer handles A first, independently toggling this native
        # flag. Our A action must clear it, rather than only clearing user_scn.
        view.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT]=True
        view._handle_custom_action(ViewerAction.TOGGLE_SHOW_ALL_ENVS,None)
        self.assertFalse(view.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT])
        # Mouse zoom does not dispatch our keyboard handlers.
        view.viewer.cam.distance=3.5
        view._update_debug_visualizers(view.viewer)
        self.assertEqual(scene.ngeom,0)
        # Native picking can switch camera type independently as well.
        view._overview=True
        view.viewer.cam.distance=2500;view.viewer.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING
        view._update_debug_visualizers(view.viewer)
        self.assertEqual(scene.ngeom,0)

        view._setup_camera()
        self.assertFalse(view._overview)
        self.assertEqual(scene.ngeom,0)

    def test_native_auto_connect_cannot_add_opaque_blue_geometry(self):
        model=self.shared.model
        self.assertEqual(float(model.vis.rgba.connect[3]),0.)
        options=mujoco.MjvOption()
        options.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT]=True
        camera=mujoco.MjvCamera();camera.lookat[:]=self.shared.data.xpos[self.shared.pelvis[0]]
        camera.distance=3.5
        scene=mujoco.MjvScene(model,maxgeom=2000)
        mujoco.mjv_updateScene(model,self.shared.data,options,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
        blue=np.array([.2,.2,.8])
        self.assertFalse(any(g.rgba[3]>0 and np.allclose(g.rgba[:3],blue) for g in scene.geoms[:scene.ngeom]))


if __name__=='__main__':unittest.main()
