"""Native static scene/distance and fake-buffer reset checks; no simulation."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import mujoco
import numpy as np
import torch
import flat_support_task as flat
from mjlab.scene import Scene


class FlatSupport(unittest.TestCase):
    def test_real_train_and_skin_scenes_have_plane_and_no_dem(self):
        for play in (False,True):
            with patch('lunar_terrain.terrain_data',side_effect=AssertionError('Flat task must not load DEM')):
                cfg=flat.terrain_stand_cfg(play)
                self.assertEqual(cfg.scene.num_envs,1 if play else 8192)
                self.assertEqual(cfg.episode_length_s,10.)
                cfg.scene.num_envs=2
                scene=Scene(cfg.scene,device='cpu');model=scene.compile()
            cfg.sim.mujoco.apply(model)
            self.assertEqual(model.nhfield,0)
            self.assertEqual(np.count_nonzero(model.geom_type==int(mujoco.mjtGeom.mjGEOM_PLANE)),1)
            self.assertEqual((model.nq,model.nv,model.nu),(30,29,23))
            self.assertEqual(model.nskin,9 if play else 0)
            self.assertAlmostEqual(model.opt.gravity[2],-1.635)
            self.assertEqual(cfg.rewards['joint_pose'].weight,-.25)
            self.assertEqual(cfg.rewards['centroidal_angular_momentum'].weight,-.01)
            self.assertNotIn('pelvis_velocity',cfg.rewards)
            self.assertEqual(cfg.rewards['torso_velocity'].weight,-4.)
            self.assertEqual(model.sensor('robot/balance_torso_com_vel').dim[0],3)
            for name,weight in flat.REWARD_WEIGHTS.items():
                self.assertEqual(cfg.rewards[name].weight,weight)
            self.assertNotIn('terrain_edge',cfg.terminations)
            self.assertIn('left_station',cfg.terminations)
            self.assertEqual(next(iter(cfg.terminations)),'numerical_guard')
            self.assertFalse(any('sole_tile' in model.geom(i).name for i in range(model.ngeom)))
            self.assertTrue(any(kind==int(mujoco.mjtSensor.mjSENS_CONTACT) for kind in model.sensor_type))
            data=mujoco.MjData(model);data.qpos[2]=flat.rest_height()
            mujoco.mj_kinematics(model,data)
            gaps=[mujoco.mj_geomDistance(model,data,model.geom('terrain').id,
                  model.geom('robot/'+side+'_sole').id,2.,None) for side in ('left','right')]
            self.assertAlmostEqual(min(gaps),.015,places=6)
            self.assertLess(max(gaps),.016)

    def test_partial_reset_sets_fixed_pose_targets_and_preserves_other_worlds(self):
        class Robot:
            def __init__(self):
                state=torch.zeros(3,13);state[:,3]=1
                self.data=SimpleNamespace(default_root_state=state,default_joint_pos=torch.zeros(3,23))
            def write_root_state_to_sim(self,state,env_ids):self.state=state.clone();self.ids=env_ids.clone()
            def write_joint_state_to_sim(self,q,v,env_ids):self.q=q.clone();self.v=v.clone()
            def set_joint_position_target(self,q,env_ids):self.target=q.clone()
        class FakeScene(dict):pass
        robot=Robot();scene=FakeScene(robot=robot)
        scene.env_origins=torch.tensor([[0.,0.,0.],[3.,0.,0.],[6.,0.,0.]])
        env=SimpleNamespace(scene=scene,num_envs=3,device='cpu',_flat_stand_verified=True,
            _stand_target_root=torch.full((3,3),42.),_stand_target_heading=torch.full((3,4),42.))
        ids=torch.tensor([1,2])
        with patch.object(flat,'reset_scene_to_default') as standard:
            flat.reset_plane(env,ids);standard.assert_called_once_with(env,ids)
        torch.testing.assert_close(robot.state[:,:2],scene.env_origins[ids,:2])
        self.assertTrue((robot.state[:,2]==flat.rest_height()).all())
        self.assertFalse(robot.state[:,7:].any());self.assertFalse(robot.v.any())
        torch.testing.assert_close(robot.q,robot.target)
        self.assertTrue((env._stand_target_root[0]==42).all())
        self.assertEqual(env._stand_reset_generation,1)


if __name__=='__main__':unittest.main()
