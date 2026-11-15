"""CPU configuration/reset checks; no environment construction or physics."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import mujoco
import torch
import lqr_flat_task as task
from lunar_terrain import FlatSurface


class FlatLqr(unittest.TestCase):
    def test_plane_surface_is_translation_invariant(self):
        surface=FlatSurface()
        torch.testing.assert_close(surface.height(torch.randn((7,5,2))),torch.zeros((7,5)))
        torch.testing.assert_close(surface.scan(torch.randn((7,7))),torch.zeros((7,25)))

    def test_reset_uses_equilibrium_without_gap_and_yaw_heading_without_lean(self):
        spec=mujoco.MjSpec()
        spec.worldbody.add_geom(name='ground',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[0,0,.01])
        task.configure_plane(spec)
        model=spec.compile()
        default=torch.zeros((3,13));default[:,3]=1.
        robot=SimpleNamespace(data=SimpleNamespace(default_root_state=default),
             write_root_state_to_sim=Mock(),write_joint_state_to_sim=Mock(),set_joint_position_target=Mock())
        class Scene(dict):pass
        scene=Scene(robot=robot);scene.env_origins=torch.tensor([[0.,0.,0.],[3.,0.,0.],[6.,0.,0.]])
        env=SimpleNamespace(scene=scene,device='cpu',num_envs=3,sim=SimpleNamespace(mj_model=model))
        qref=torch.zeros(30);qref[2]=.81;qref[3]=.99955;qref[4]=.03
        qref[7:]=torch.arange(23)*.001
        controller=SimpleNamespace(qref=qref,ctrl0=torch.arange(23)*.002)
        with patch.object(task,'reset_scene_to_default'),patch.object(task,'term',return_value=controller):
            task.reset_equilibrium(env,torch.tensor([1]))
        state=robot.write_root_state_to_sim.call_args.args[0]
        torch.testing.assert_close(state[0,:3],torch.tensor([3.,0.,.81]))
        torch.testing.assert_close(state[0,3:7],qref[3:7])
        self.assertFalse(state[0,7:].any())
        joint=robot.write_joint_state_to_sim.call_args.args[0]
        torch.testing.assert_close(joint[0],qref[7:])
        torch.testing.assert_close(env._stand_target_heading[1],torch.tensor([1.,0.,0.,0.]))
        torch.testing.assert_close(robot.set_joint_position_target.call_args.args[0][0],controller.ctrl0)

    def test_config_has_no_hfield_reset_or_edge(self):
        for play in (False,True):
            cfg=task.terrain_stand_cfg(play)
            self.assertEqual(cfg.scene.terrain.terrain_type,'plane')
            self.assertIs(cfg.events['reset_pose'].func,task.reset_equilibrium)
            self.assertNotIn('terrain_edge',cfg.terminations)
            self.assertIsInstance(cfg.actions['position'],task.EquilibriumLqrActionCfg)


if __name__=='__main__':unittest.main()
