"""Static native DEM bounds and interpolation checks; no simulation."""
from types import SimpleNamespace
import unittest,warnings
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from lunar_terrain import Surface
import local_lunar_terrain as terrain
import lunar_finetune_task as task


class LocalLunar(unittest.TestCase):
    def test_bank_preserves_dem_and_native_bounds(self):
        cfg=task.terrain_stand_cfg();cfg.scene.num_envs=4
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');scene=Scene(cfg.scene,'cpu');model=scene.compile()
        env=SimpleNamespace(sim=SimpleNamespace(mj_model=model),num_envs=4,device='cpu')
        ground=terrain.LocalSurface(env);ground.patch_ids[:]=torch.tensor([0,3,20,63])
        source=Surface('cpu');bank=terrain.bank_data()
        torch.manual_seed(42);xy=6+torch.randint(0,13,(4,25,2))*.1
        expected_xy=xy+torch.tensor(bank['centers'][[0,3,20,63]],dtype=torch.float32)[:,None,:]
        world=expected_xy.numpy().astype(np.float64)
        expected=terrain.base_height(world)
        for i in range(4):expected[i]+=terrain.lunar_details.generate(world[i],[0,0])-bank['altitudes'][[0,3,20,63][i]]
        expected=torch.tensor(expected,dtype=torch.float32)
        torch.testing.assert_close(ground.height(xy),expected,atol=1e-4,rtol=0)
        middle=bank['detail'].shape[1]//2
        torch.testing.assert_close(ground.height(torch.zeros(4,2)),
            torch.tensor(bank['detail'][[0,3,20,63],middle,middle],dtype=torch.float32),atol=1e-6,rtol=0)
        # Native ray-to-hfield query independently checks triangle orientation,
        # per-asset scale and per-world vertical offset; no time integration.
        model.geom_group[:]=0;model.geom_group[ground.geom_id]=5
        data=mujoco.MjData(model)
        for world,selection in enumerate((0,3,20,63)):
            model.geom_dataid[ground.geom_id]=int(ground.native_ids[selection])
            model.geom_pos[ground.geom_id,2]=bank['zmin'][selection]
            mujoco.mj_kinematics(model,data)
            point=np.array([.55,-.45,10.]);direction=np.array([0.,0.,-1.])
            distance=mujoco.mj_ray(model,data,point,direction,np.array([0,0,0,0,0,1],dtype=np.uint8),True,-1,None)
            height=float(ground.height(torch.tensor([[.55,-.45]]),torch.tensor([world]))[0])
            self.assertAlmostEqual(10-distance,height,places=5)
        self.assertEqual(model.nhfield,64)
        # The native GPU kinematics kernel skips world-welded static geoms.
        # A mocap ancestor is required for per-world geom_pos resets to refresh
        # the actual collision pose (not just the CPU visualization model).
        body=int(model.geom_bodyid[ground.geom_id])
        self.assertGreaterEqual(int(model.body_mocapid[int(model.body_rootid[body])]),0)
        self.assertEqual(model.nv,29)  # six floating root + 23 joints, no terrain DOF
        self.assertEqual(task.OBS_DIM,146)
        self.assertEqual(task.REWARD_WEIGHTS['torso_velocity'],-4.)
        self.assertNotIn('capture_risk',cfg.rewards)
        self.assertEqual(cfg.events['reset_pose'].func.model_fields,('geom_dataid','geom_pos'))
        np.testing.assert_array_equal(scene.env_origins,0.)

    def test_reset_native_index_dtype_and_all_spawn_bounds(self):
        cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=4
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');scene=Scene(cfg.scene,'cpu');model=scene.compile()
        n=256
        class Robot:
            data=SimpleNamespace(default_root_state=torch.zeros(n,13),default_joint_pos=torch.zeros(n,23))
            def write_root_state_to_sim(self,state,env_ids):self.state=state
            def write_joint_state_to_sim(self,q,v,env_ids):pass
            def set_joint_position_target(self,q,env_ids):pass
        class FakeScene(dict):pass
        fake=FakeScene(robot=Robot());fake.env_origins=torch.zeros(n,3)
        fields=SimpleNamespace(geom_dataid=torch.tensor(model.geom_dataid).repeat(n,1),
                               geom_pos=torch.tensor(model.geom_pos,dtype=torch.float32).repeat(n,1,1))
        env=SimpleNamespace(sim=SimpleNamespace(mj_model=model,model=fields),num_envs=n,device='cpu',scene=fake)
        with patch.object(task,'reset_scene_to_default'):task.reset_local(env,None)
        self.assertTrue(torch.isfinite(fake['robot'].state).all())
        self.assertLess(float(fake['robot'].state[:,:2].abs().max()),1.251)
        self.assertTrue((fields.geom_dataid[:,env._local_lunar_surface.geom_id]>=0).all())
        ground=env._local_lunar_surface
        torch.testing.assert_close(fields.geom_pos[:,ground.visual_ids],ground.visual_offsets[ground.patch_ids])
        small=SimpleNamespace(sim=env.sim,num_envs=16,device='cpu',scene=fake)
        with patch.object(task,'reset_scene_to_default'):
            task.reset_local(small,None,distinct_patches=True)
            self.assertEqual(small._local_lunar_surface.patch_ids.unique().numel(),16)
            task.reset_local(small,torch.tensor([1,5,8]),distinct_patches=True)
            self.assertEqual(small._local_lunar_surface.patch_ids.unique().numel(),16)

    def test_fused_spawn_guard_still_rejects_both_failure_modes(self):
        """One host reduction now covers both spawn guards; each must still raise."""
        cfg=task.terrain_stand_cfg();cfg.scene.num_envs=4
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');scene=Scene(cfg.scene,'cpu');model=scene.compile()
        n=8
        class Robot:
            data=SimpleNamespace(default_root_state=torch.zeros(n,13),default_joint_pos=torch.zeros(n,23))
            def write_root_state_to_sim(self,state,env_ids):pass
            def write_joint_state_to_sim(self,q,v,env_ids):pass
            def set_joint_position_target(self,q,env_ids):pass
        class FakeScene(dict):pass
        fake=FakeScene(robot=Robot());fake.env_origins=torch.zeros(n,3)
        fields=SimpleNamespace(geom_dataid=torch.tensor(model.geom_dataid).repeat(n,1),
                               geom_pos=torch.tensor(model.geom_pos,dtype=torch.float32).repeat(n,1,1))
        env=SimpleNamespace(sim=SimpleNamespace(mj_model=model,model=fields),num_envs=n,device='cpu',scene=fake)
        with patch.object(task,'reset_scene_to_default'):
            task.reset_local(env,None)  # Builds the surface; unmodified spawn passes.
            ground=env._local_lunar_surface
            # Soles pushed outside the 2m slope-certified region.
            with patch.object(ground,'spawn_xy',ground.spawn_xy+5.):
                with self.assertRaisesRegex(RuntimeError,'certified slope-bound region'):
                    task.reset_local(env,None)
            # Certified region intact, but the sole clearance is invalid.
            with patch.object(terrain,'CLEARANCE',-1.):
                with self.assertRaisesRegex(RuntimeError,'Invalid local terrain spawn'):
                    task.reset_local(env,None)

    def test_full_original_dem_view_matches_local_collision_heights(self):
        cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=4
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');scene=Scene(cfg.scene,'cpu');model=scene.compile()
        ground=terrain.LocalSurface(SimpleNamespace(sim=SimpleNamespace(mj_model=model),num_envs=4,device='cpu'))
        vid=ground.visual_id;self.assertIsNotNone(vid)
        self.assertEqual(model.nhfield,64)
        self.assertEqual(int(model.geom_contype[vid]),0)
        self.assertEqual(int(model.geom_conaffinity[vid]),0)
        self.assertEqual(model.geom_type[vid],mujoco.mjtGeom.mjGEOM_MESH)
        # Independent native ray against FULL visual DEM, compared to sampled
        # local collision heights at the same points. No mj_step or GPU run.
        selections=torch.tensor([0,3,20,63]);ground.patch_ids[:]=selections
        model.geom_group[:]=0;model.geom_group[ground.visual_ids.numpy()]=5
        data=mujoco.MjData(model)
        for world,selection in enumerate(selections.tolist()):
            model.geom_pos[ground.visual_ids.numpy()]=ground.visual_offsets[selection].numpy()
            mujoco.mj_kinematics(model,data)
            for x,y in ((.553,-.431),(-1.117,.823),(.017,.033),(7.999,.137),(-7.999,2.513),(1.337,7.999),(-2.537,-7.999)):
                distance=mujoco.mj_ray(model,data,np.array([x,y,10.]),np.array([0.,0.,-1.]),
                                      np.array([0,0,0,0,0,1],dtype=np.uint8),True,-1,None)
                expected=float(ground.height(torch.tensor([[x,y]],dtype=torch.float32),torch.tensor([world]))[0])
                self.assertAlmostEqual(10-distance,expected,delta=1e-4)
            # Beyond the insert the background still uses the original DEM.
            x,y=6.117,-6.329
            distance=mujoco.mj_ray(model,data,np.array([x,y,10.]),np.array([0.,0.,-1.]),
                                  np.array([0,0,0,0,0,1],dtype=np.uint8),True,-1,None)
            expected=float(ground.height(torch.tensor([[x,y]],dtype=torch.float32),torch.tensor([world]))[0])
            self.assertAlmostEqual(10-distance,expected,delta=1e-4)


if __name__=='__main__':unittest.main()
