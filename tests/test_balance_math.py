"""Pure CPU tensor checks; no environment construction, physics steps or training.
Run explicitly: PYTHONPATH=src <shared-python> -m unittest discover -s tests
"""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
from balance_mdp import features, SENSORS, SENSOR_DIM, OBS_DIM, INDEX, JOINT_DIM, STATE_DIM

ROOT=Path(__file__).resolve().parents[1]
WEIGHT=126.8*9.81/6


def weights_from_source():
    tree=ast.parse((ROOT/'src/balance_task.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REWARD_WEIGHTS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('Missing declared reward weights')


def state():
    x=torch.zeros(1,STATE_DIM+SENSOR_DIM+30,dtype=torch.float64)
    spans={};offset=STATE_DIM
    for name,_,_,_,width in SENSORS:
        spans[name]=slice(offset,offset+width);offset+=width
    x[:,spans['root_pos']]=torch.tensor([0,0,.81])
    for name in ('root_quat','torso_quat'): x[:,spans[name]]=torch.tensor([1,0,0,0])
    x[:,spans['com_pos']]=torch.tensor([-.02,0,1.])
    x[:,spans['left_pos']]=torch.tensor([.035,.17,0.])
    x[:,spans['right_pos']]=torch.tensor([.035,-.17,0.])
    x[:,spans['left_load']]=WEIGHT/2;x[:,spans['right_load']]=WEIGHT/2
    return x,spans


class BalanceMath(unittest.TestCase):
    def test_standard_torque_cost_uses_actual_actuator_force(self):
        from mjlab.envs.mdp.rewards import joint_torques_l2
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        # Unit transmission gear: native actuator force is actual joint Nm.
        forces=torch.zeros((2,23));forces[0,:2]=torch.tensor([3.,-4.]);forces[1,:]=10.
        env=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(actuator_force=forces))})
        value=joint_torques_l2(env,SceneEntityCfg('robot'))
        torch.testing.assert_close(value,torch.tensor([25.,2300.]))
        self.assertAlmostEqual(float(value[1])*weights_from_source()['joint_torques'],-.0046)

    def test_dimensions_and_airborne_are_finite(self):
        x,s=state();x[:,s['left_load']]=0;x[:,s['right_load']]=0
        result=features(x,WEIGHT,9.81/6)
        self.assertEqual(result.shape,(1,OBS_DIM+len(INDEX)))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result[0,INDEX['double_contact']],0)

    def test_native_momentum_and_pelvis_velocity_remain_separate(self):
        x,s=state();x[:,s['root_vel']]=torch.tensor([.1,.2,.3])
        x[:,s['com_vel']]=torch.tensor([1.,2.,3.])
        x[:,s['momentum']]=torch.tensor([2.,3.,6.])
        r=features(x,WEIGHT,9.81/6)
        self.assertAlmostEqual(float(r[0,INDEX['pelvis_speed_squared']]),.14,places=6)
        self.assertAlmostEqual(float(r[0,INDEX['momentum_squared']]),49.,places=6)

    def test_observation_invariant_to_world_translation(self):
        x,s=state();reference=features(x,WEIGHT,9.81/6)
        for name in ('root_pos','com_pos','left_pos','right_pos'):
            x[:,s[name]]+=torch.tensor([300.,-400.,-200.])
        x[:,STATE_DIM+SENSOR_DIM+2:STATE_DIM+SENSOR_DIM+5]-=200.
        torch.testing.assert_close(features(x,WEIGHT,9.81/6),reference,atol=1e-10,rtol=1e-10)

    def test_angular_velocity_exposes_tilt_growth_immediately(self):
        x,s=state();x[:,s['root_omega']]=torch.tensor([0.,.1,0.])
        result=features(x,WEIGHT,9.81/6)
        self.assertGreater(float(result[0,2*JOINT_DIM+45]),0.)
        x[:,s['root_omega']]=torch.tensor([0.,-.1,0.])
        self.assertLess(float(features(x,WEIGHT,9.81/6)[0,2*JOINT_DIM+45]),0.)

    def test_joint_pose_penalty_covers_all_axes_and_is_not_an_action_penalty(self):
        x,_=state()
        self.assertEqual(JOINT_DIM,23)
        self.assertEqual(OBS_DIM,145)
        self.assertEqual(float(features(x,WEIGHT,9.81/6)[0,INDEX['joint_pose_squared']]),0.)
        for joint in range(JOINT_DIM):
            y=x.clone();y[:,joint]=.2
            self.assertAlmostEqual(float(features(y,WEIGHT,9.81/6)[0,INDEX['joint_pose_squared']]),.04/23)
        x[:,2*JOINT_DIM:3*JOINT_DIM]=1.
        self.assertEqual(float(features(x,WEIGHT,9.81/6)[0,INDEX['joint_pose_squared']]),0.)
        self.assertLess(weights_from_source()['joint_pose'],0.)

    def test_weight_budget_hypothetical_unstable_vs_stable(self):
        # Analytic examples, NOT evidence of actual training reward signs.
        w=weights_from_source()
        def rate(speed,momentum,tilt):
            return w['alive']+w['pelvis_velocity']*speed**2+w['centroidal_angular_momentum']*momentum**2+w['gravity_tilt']*tilt**2
        self.assertTrue(-.15<rate(.2,5.,torch.pi/18)<0)
        self.assertGreater(rate(.02,1.,torch.pi/60),0)

if __name__=='__main__': unittest.main()
