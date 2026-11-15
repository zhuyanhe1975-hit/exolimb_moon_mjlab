"""CPU tensor/config tests only; no simulation, training or viewer."""
import unittest
import torch
from support_mdp import signed_margin,compute_geometry,geometry_input


class SupportMargin(unittest.TestCase):
    def test_square_inside_outside_and_boundary(self):
        p=torch.tensor([[[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]]).expand(4,-1,-1)
        q=torch.tensor([[0.,0.],[2.,0.],[2.,2.],[1.,0.]])
        m,area=signed_margin(p,torch.ones((4,4),dtype=torch.bool),q)
        torch.testing.assert_close(m,torch.tensor([1.,-1.,-2.**.5,0.]))
        self.assertTrue(area.all())

    def test_empty_point_line_and_duplicates_never_positive(self):
        p=torch.tensor([[[0.,0.],[0.,0.],[1.,0.],[2.,0.]]]).expand(3,-1,-1)
        mask=torch.tensor([[False]*4,[True,False,False,False],[True]*4])
        q=torch.tensor([[100.,100.],[0.,1.],[1.,.5]])
        m,area=signed_margin(p,mask,q)
        torch.testing.assert_close(m,torch.tensor([-.2,-1.,-.5]))
        self.assertFalse(area.any())

    def test_lunar_velocity_predicts_exit_before_static_com(self):
        p=torch.tensor([[[-.1,-.1,0.],[.1,-.1,0.],[.1,.1,0.],[-.1,.1,0.]]])
        valid=torch.ones((1,4),dtype=torch.bool)
        c=torch.tensor([[0.,0.,1.]])
        v=torch.tensor([[.2,0.,0.]])
        lunar=compute_geometry(p,valid,c,v,1.635)
        earth=compute_geometry(p,valid,c,v,9.81)
        self.assertGreater(lunar[0,0],0)
        self.assertLess(lunar[0,1],0)
        self.assertGreater(earth[0,1],0)
        self.assertLess(lunar[0,1],earth[0,1])

    def test_world_translation_and_contact_order_invariance(self):
        p=torch.tensor([[[-.1,-.1,0.],[.1,-.1,0.],[.1,.1,0.],[-.1,.1,0.]]],dtype=torch.float64)
        valid=torch.ones((1,4),dtype=torch.bool)
        c=torch.tensor([[0.,0.,1.]],dtype=torch.float64);v=torch.tensor([[.2,.1,0.]],dtype=torch.float64)
        a=compute_geometry(p,valid,c,v,1.635)
        offset=torch.tensor([900.,-800.,100.],dtype=torch.float64)
        b=compute_geometry((p+offset)[:,[2,0,3,1]],valid,c+offset,v,1.635)
        torch.testing.assert_close(a,b)

    def test_actor_capture_vector_uses_heading_frame(self):
        points=torch.tensor([[[-.1,-.1,0.],[.1,-.1,0.],[.1,.1,0.],[-.1,.1,0.]]]).repeat(1,2,1)
        valid=torch.ones((1,8,1))
        packed=torch.cat((torch.cat((points,valid),-1).flatten(1),
                          torch.tensor([[0.,0.,1.,.2,0.,0.,2**-.5,0.,0.,2**-.5]])),-1)
        result=geometry_input(packed,1.635)
        self.assertAlmostEqual(result[0,2].item(),0.,places=6)
        self.assertLess(result[0,3],0)

    def test_v10_weights_and_sensor_configuration(self):
        import support_task as task
        for play in (False,True):
            c=task.terrain_stand_cfg(play)
            for name in ('gravity_tilt',):
                self.assertEqual(c.rewards[name].weight,0)
            self.assertEqual(c.rewards['joint_pose'].weight,-.25)
            self.assertEqual(c.rewards['centroidal_angular_momentum'].weight,-.002)
            for name,weight in task.REWARD_WEIGHTS.items():
                self.assertEqual(c.rewards[name].weight,weight)
            self.assertEqual(c.scene.sensors[-1].num_slots,4)
            self.assertEqual(c.scene.sensors[-1].reduce,'maxforce')
            self.assertEqual(c.episode_length_s,30.)
            self.assertEqual(task.OBS_DIM,151)
            self.assertEqual(c.scene.num_envs,1 if play else 8192)


if __name__=='__main__':unittest.main()
