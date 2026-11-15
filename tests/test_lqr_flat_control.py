"""No dynamics: equilibrium invariance, continuous blending and physical bounds."""
import unittest
import torch
from lqr_flat_control import equilibrium_commands


class FlatControl(unittest.TestCase):
    def call(self,dx,authority,residual=None):
        if residual is None:residual=torch.zeros((len(dx),23),dtype=torch.float64)
        return equilibrium_commands(dx,torch.eye(23,55,dtype=torch.float64),
            torch.linspace(-.1,.1,23,dtype=torch.float64),authority,residual,
            torch.full((23,),-.5,dtype=torch.float64),torch.full((23,),.5,dtype=torch.float64))

    def test_gate_changes_do_not_move_the_equilibrium_pd_target(self):
        a=torch.tensor([0.,.01,.3,.8,1.],dtype=torch.float64)
        _,final,_,_=self.call(torch.zeros((5,55),dtype=torch.float64),a)
        ref=torch.linspace(-.1,.1,23,dtype=torch.float64).expand(5,-1)
        torch.testing.assert_close(final,ref,atol=0,rtol=0)

    def test_full_authority_is_original_lqr_and_zero_retains_feedforward(self):
        dx=torch.zeros((2,55),dtype=torch.float64);dx[:,:23]=.02
        _,final,_,_=self.call(dx,torch.tensor([0.,1.],dtype=torch.float64))
        ref=torch.linspace(-.1,.1,23,dtype=torch.float64)
        torch.testing.assert_close(final[0],ref)
        torch.testing.assert_close(final[1],ref-.02)

    def test_authority_has_zero_endpoint_slope_and_final_targets_are_bounded(self):
        dx=torch.ones((2,55),dtype=torch.float64)*.01
        _,final,a,_=self.call(dx,torch.tensor([0.,1e-5],dtype=torch.float64))
        self.assertLess(float((final[1]-final[0]).abs().max()),1e-10)
        self.assertGreater(float(a[1]),0.)
        _,final,_,clipped=self.call(torch.ones((2,55),dtype=torch.float64)*10.,torch.ones(2,dtype=torch.float64))
        self.assertTrue((final.abs()<=.5).all())
        self.assertTrue((clipped==1.).all())


if __name__=='__main__':unittest.main()
