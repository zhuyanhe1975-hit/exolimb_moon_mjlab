"""CPU math/API checks only: no env creation, physics stepping or gain search."""
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch
import warp as wp

from lqr_artifact import KEEP, STATE_DIM, SCHEMA, source_digest, load_artifact
from lqr_action import tangent_state, blend_commands, ResidualLqrAction


class LqrMath(unittest.TestCase):
    def test_native_tangent_matches_differentiatepos_after_random_world_yaw(self):
        # Compile only; mj_differentiatePos/mju_* are native coordinate math,
        # not forward dynamics, simulation or controller preparation.
        from wangyu23 import restore_23_axes
        root=Path(__file__).resolve().parents[1]
        model=restore_23_axes(mujoco.MjSpec.from_file(str(root/'assets/wangyu_physics/wangyu.xml'))).compile()
        reference=model.qpos0.copy()
        mujoco.mju_quatIntegrate(reference[3:7],np.array([.03,-.04,.01]),1.)
        q=reference.copy()
        q[2]+=.02
        q[7:]+=np.linspace(-.05,.05,23)
        mujoco.mju_quatIntegrate(q[3:7],np.array([-.02,.05,-.03]),1.)
        dq=np.zeros(29)
        mujoco.mj_differentiatePos(model,dq,1.,reference,q)
        velocity=np.linspace(-.1,.1,29)
        expected=np.r_[dq,velocity]
        expected[:2]=0
        mujoco.mju_rotVecQuat(expected[3:6],dq[3:6],reference[3:7])
        mujoco.mju_rotVecQuat(expected[32:35],velocity[3:6],q[3:7])
        yaw=.91
        heading=np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
        original=q[3:7].copy()
        mujoco.mju_mulQuat(q[3:7],heading,original)
        q[:2]=[800.,-700.]
        q[2]+=400.
        v=velocity.copy()
        mujoco.mju_rotVecQuat(v[:3],velocity[:3],heading)
        arrays=[torch.tensor(a,dtype=torch.float32) for a in
                (q[None,:],v[None,:],heading[None,:],np.array([400.]),reference)]
        output=torch.zeros((1,58))
        wp.init()
        wp.launch(tangent_state,dim=1,inputs=[wp.from_torch(a) for a in arrays],
                  outputs=[wp.from_torch(output)],device='cpu')
        np.testing.assert_allclose(output.numpy()[0],expected,atol=3e-5,rtol=1e-5)
        self.assertEqual(STATE_DIM,55)
        self.assertNotIn(0,KEEP);self.assertNotIn(1,KEEP);self.assertNotIn(5,KEEP)

    def blend(self,dx=None,load=1.,slope=0.,residual=0.,control=.1):
        if dx is None: dx=torch.zeros((1,STATE_DIM))
        return blend_commands(dx,torch.zeros((23,STATE_DIM)),torch.full((23,),control),
            torch.zeros((1,23)),torch.tensor([[load/2,load/2]]),torch.tensor([slope]),
            torch.full((1,23),residual),torch.full((23,),-.5),torch.full((23,),.5),.05)

    def test_zero_residual_is_teacher_and_gate_drops_without_support(self):
        base,final,alpha,_=self.blend()
        torch.testing.assert_close(base,final)
        self.assertEqual(float(alpha[0]),1.)
        for args in ({'load':0.},{'slope':np.deg2rad(15.)}):
            _,final,alpha,_=self.blend(**args)
            self.assertEqual(float(alpha[0]),0.)
            self.assertEqual(float(final.abs().sum()),0.)
        dx=torch.zeros((1,55));dx[:,1]=.5
        self.assertEqual(float(self.blend(dx=dx)[2][0]),0.)
        # Single support must not use a double-support gain at full authority.
        _,_,alpha,_=blend_commands(torch.zeros((1,55)),torch.zeros((23,55)),
            torch.ones(23)*.1,torch.zeros((1,23)),torch.tensor([[1.,0.]]),torch.zeros(1),
            torch.zeros((1,23)),torch.full((23,),-.5),torch.full((23,),.5),.05)
        self.assertEqual(float(alpha[0]),0.)

    def test_residual_limits_and_final_target_clipping(self):
        _,final,_,saturation=self.blend(residual=100.,control=.45)
        torch.testing.assert_close(final,torch.full((1,23),.5))
        self.assertEqual(float(saturation[0]),1.)
        # Baseline clipping must also be reported, even with no RL residual.
        self.assertEqual(float(self.blend(control=1.)[3][0]),1.)
        _,final,alpha,_=self.blend(load=0.,residual=2.)
        self.assertEqual(float(alpha[0]),0.)
        torch.testing.assert_close(final,torch.full((1,23),.5))

    def test_partial_reset_clears_both_final_command_history_buffers(self):
        obj=ResidualLqrAction.__new__(ResidualLqrAction)
        obj._offset=torch.zeros((3,23))
        for name in ('_raw_actions','last_command','previous_command','_processed_actions','baseline'):
            setattr(obj,name,torch.ones((3,23)))
        for name in ('applied_alpha','applied_saturation','preview_alpha'):
            setattr(obj,name,torch.ones(3))
        obj.reset(torch.tensor([1]))
        for name in ('_raw_actions','last_command','previous_command','_processed_actions','baseline'):
            value=getattr(obj,name)
            self.assertEqual(float(value[1].sum()),0.)
            self.assertEqual(float(value[0].sum()),23.)
        self.assertEqual(float(obj.applied_alpha[1]),0.)

    def test_artifact_rejects_nan_and_stale_source(self):
        names=np.array([f'joint_{i}' for i in range(23)])
        metadata={'schema':SCHEMA,'source_sha256':source_digest(),'validated':True,
                  'spectral_radius':.99,'gravity':9.81/6,'physics_dt':1/120,'decimation':2}
        q=np.zeros(30);q[3]=1
        payload=dict(K=np.zeros((23,55)),qpos0=q,ctrl0=np.zeros(23),keep=KEEP,
                     joint_names=names,actuator_joints=names,kp=np.ones(23),kd=np.ones(23),
                     force_limit=np.ones(23),joint_range=np.tile([-1.,1.],(23,1)),body_mass=np.ones(14))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.npz'
            def write(): np.savez(path,metadata=np.array(json.dumps(metadata)),**payload)
            write();self.assertEqual(load_artifact(path)[0]['K'].shape,(23,55))
            payload['K'][0,0]=np.nan;write()
            with self.assertRaisesRegex(ValueError,'Nonfinite'): load_artifact(path)
            payload['K'][0,0]=0;metadata['source_sha256']='outdated';write()
            with self.assertRaisesRegex(ValueError,'source mismatch'): load_artifact(path)


if __name__=='__main__': unittest.main()
