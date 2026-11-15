"""CPU native kinematics + G1/MJLab gravity observation; no dynamics or policy."""
from types import SimpleNamespace
from unittest.mock import patch
import math
import warnings
import unittest
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.entity.data import EntityData
import balance_mdp
import no_margin_mdp
import torso_observation_mdp as mdp
import flat_torso_observation_task as task


class TorsoObservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg=task.terrain_stand_cfg();cfg.scene.num_envs=1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');cls.scene=Scene(cfg.scene,'cpu');cls.model=cls.scene.compile()

    def test_native_motion_and_projected_gravity_match_mujoco_rotation(self):
        m=self.model;d=mujoco.MjData(m)
        angle=.6
        d.qpos[3:7]=[math.cos(angle/2),0,0,math.sin(angle/2)]
        d.qvel[:3]=[.1,-.2,.05]
        d.qvel[int(m.jnt_dofadr[m.joint('robot/WaistPitch').id])]=.4
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        mujoco.mj_comVel(m,d);mujoco.mj_sensorPos(m,d);mujoco.mj_sensorVel(m,d)
        root=m.body('robot/Pelvis').id
        proxy=SimpleNamespace(root_link_quat_w=torch.tensor(d.xquat[root])[None],
                              gravity_vec_w=torch.tensor([[0.,0.,-1.]],dtype=torch.float64))
        gravity=EntityData.projected_gravity_b.fget(proxy)
        native=SimpleNamespace(sensordata=torch.tensor(d.sensordata)[None])
        env=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))},
                            sim=SimpleNamespace(mj_model=m,data=native))
        raw=[torch.zeros(1,balance_mdp.STATE_DIM,dtype=torch.float64)]
        for name,_,_,_,width in balance_mdp.SENSORS:
            a=int(m.sensor('robot/balance_'+name).adr[0]);raw.append(native.sensordata[:,a:a+width])
        raw.append(torch.zeros(1,30,dtype=torch.float64))
        legacy=balance_mdp.features(torch.cat(raw,-1),207.318,1.635)
        state=no_margin_mdp.without_prediction(legacy)
        with patch.object(balance_mdp,'observation',return_value=legacy[:,:balance_mdp.OBS_DIM]):result=mdp.observation(env)
        expected=torch.cat((state[:,:46],gravity,state[:,49:],mdp.local_torso_velocity(env)/.3),-1)
        torch.testing.assert_close(result,expected,rtol=0,atol=0)
        self.assertEqual(result.shape,(1,146))
        expected_gravity=d.xmat[root].reshape(3,3).T@np.array([0.,0.,-1.])
        np.testing.assert_allclose(result[0,mdp.ROOT_GRAVITY],expected_gravity,atol=1e-12)
        # Same root-gravity convention as before, exactly the standard G1 term.
        torch.testing.assert_close(result[:,46:49],state[:,46:49])
        a=int(m.sensor('robot/balance_torso_com_vel').adr[0])
        world=d.sensordata[a:a+3];torso=m.body('robot/Torso').id
        expected_linear=d.xmat[torso].reshape(3,3).T@world/.3
        np.testing.assert_allclose(result[0,mdp.TORSO_LINEAR_VELOCITY],expected_linear,atol=1e-12)
        a=int(m.sensor('robot/balance_torso_omega').adr[0])
        expected_angular=d.xmat[torso].reshape(3,3).T@d.sensordata[a:a+3]*.5
        np.testing.assert_allclose(result[0,mdp.TORSO_ANGULAR_VELOCITY],expected_angular,atol=1e-12)
        torch.testing.assert_close(result[:,49:143],state[:,49:])
        self.assertEqual(d.time,0.)

    def test_gravity_direction_roll_pitch_yaw_and_unit_length(self):
        quats=torch.tensor([[1,0,0,0],[2**-.5,2**-.5,0,0],
                            [2**-.5,0,2**-.5,0],[2**-.5,0,0,2**-.5]],dtype=torch.float64)
        proxy=SimpleNamespace(root_link_quat_w=quats,gravity_vec_w=torch.tensor([[0.,0.,-1.]]*4,dtype=torch.float64))
        actual=EntityData.projected_gravity_b.fget(proxy)
        expected=torch.tensor([[0.,0.,-1.],[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]],dtype=torch.float64)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(actual.norm(dim=-1),torch.ones(4,dtype=torch.float64))

    def test_actor_critic_146_keep_rewards_and_no_margin(self):
        for play in (False,True):
            cfg=task.terrain_stand_cfg(play)
            self.assertEqual(cfg.scene.num_envs,1 if play else 8192)
            self.assertEqual(cfg.episode_length_s,10.)
            self.assertNotIn('capture_risk',cfg.rewards)
            self.assertEqual(cfg.rewards['torso_velocity'].weight,-4.)
            self.assertEqual(cfg.rewards['torso_angular_velocity'].weight,-.2)
            self.assertEqual(cfg.rewards['centroidal_angular_momentum'].weight,-.01)
            for group in ('actor','critic'):
                fn=cfg.observations[group].terms['state'].func
                self.assertIs(fn,mdp.guarded_observation)
                self.assertEqual(fn(SimpleNamespace(num_envs=2,device='cpu')).shape,(2,146))


if __name__=='__main__':unittest.main()
