"""Real ContactSensor over prescribed CPU buffers; no forward/step/GPU/policy."""
from types import SimpleNamespace
from unittest.mock import patch
import warnings
import unittest
import torch
from mjlab.scene import Scene
import flat_support_task
import support_mdp as mdp


class SupportContactPhase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg=flat_support_task.terrain_stand_cfg();cfg.scene.num_envs=2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');cls.scene=Scene(cfg.scene,device='cpu');cls.model=cls.scene.compile()

    def setUp(self):
        self.sensor=self.scene['support_contacts']
        self.buffer=SimpleNamespace(sensordata=torch.zeros(2,self.model.nsensordata,dtype=torch.float64))
        self.sensor.initialize(self.model,None,self.buffer,'cpu');self.sensor.reset()
        self.env=SimpleNamespace(scene=self.scene,device='cpu',num_envs=2,common_step_counter=1,
            sim=SimpleNamespace(mj_model=self.model,data=self.buffer),
            cfg=SimpleNamespace(sim=SimpleNamespace(mujoco=SimpleNamespace(gravity=(0,0,-1.635)))))
        self.points=torch.tensor([[-.15,.05,0.],[.15,.05,0.],[-.15,.2,0.],[.15,.2,0.],
                                  [-.15,-.05,0.],[.15,-.05,0.],[-.15,-.2,0.],[.15,-.2,0.]],dtype=torch.float64)
        self.write_contacts();self.write_native('com_pos',[0,0,1])
        self.write_native('com_vel',[0,0,0]);self.write_native('root_quat',[1,0,0,0])

    def write_native(self,name,value):
        s=self.model.sensor('robot/balance_'+name);a=int(s.adr[0]);n=int(s.dim[0])
        self.buffer.sensordata[:,a:a+n]=torch.tensor(value,dtype=torch.float64)

    def write_contacts(self,shift=0.,airborne=False,world=None):
        ids=slice(None) if world is None else world
        for i,primary in enumerate(self.sensor.primary_names):
            positions=self.points[4*i:4*i+4].clone();positions[:,0]+=shift
            force=torch.tensor([[20.,0.,0.]]*4);force.zero_() if airborne else None
            fields={'found':torch.zeros(4) if airborne else torch.full((4,),4.),
                    'dist':torch.zeros(4) if airborne else torch.full((4,),-.001),
                    'force':force,'pos':positions}
            for name,value in fields.items():
                sensor=self.model.sensor(f'support_contacts_{primary}_{name}')
                a=int(sensor.adr[0]);n=int(sensor.dim[0])
                self.buffer.sensordata[ids,a:a+n]=value.flatten()

    def evaluate(self,phase):
        # Base actor bundle is unrelated to this native contact cache contract.
        with patch.object(mdp.base,'data',return_value=torch.empty(0)):
            return mdp.data(self.env,phase)

    def test_takeoff_observation_cannot_reuse_loaded_reward_contacts(self):
        reward=self.evaluate('reward').clone()
        self.assertTrue((reward[:,1]>0).all())
        self.write_contacts(airborne=True);self.write_native('com_pos',[0,0,1.2])
        self.assertTrue((self.sensor.data.found>0).any())  # reproduce the upstream cache
        with patch.object(self.sensor,'update',wraps=self.sensor.update) as update:
            observation=self.evaluate('observation')
            self.assertTrue((observation[:,1]==-.2).all())
            self.assertFalse(observation[:,4:].any())
            self.evaluate('observation')  # actor/critic share the same phase result
            update.assert_called_once_with(dt=0.)
        self.assertTrue((reward[:,1]>0).all())  # pre-forward result remains owned

    def test_new_com_and_new_contact_positions_are_paired(self):
        before=self.evaluate('reward').clone()
        self.write_contacts(shift=.3);self.write_native('com_pos',[.3,0,1])
        after=self.evaluate('observation')
        torch.testing.assert_close(before,after)

    def test_next_physics_step_reward_uses_native_sensor_update(self):
        self.evaluate('observation')
        self.env.common_step_counter+=1;self.write_contacts(airborne=True)
        self.sensor.update(1/120)  # normal MJLab scene.update in the physics loop
        with patch.object(self.sensor,'update',wraps=self.sensor.update) as update:
            value=self.evaluate('reward');self.assertFalse(value[:,4:].any())
            update.assert_not_called()  # no extra sensor-time advance in reward

    def test_partial_reset_generation_refreshes_same_control_step(self):
        self.evaluate('observation')
        self.env._stand_reset_generation=1;self.write_contacts(airborne=True,world=0)
        self.sensor.reset(torch.tensor([0]))
        value=self.evaluate('observation')
        self.assertEqual(value[0,1],-.2);self.assertGreater(value[1,1],0)

    def test_stateful_sensor_is_rejected_without_pushing_history(self):
        with patch.object(self.sensor.cfg,'history_length',2),patch.object(self.sensor,'update') as update:
            with self.assertRaisesRegex(ValueError,'stateless'):self.evaluate('observation')
            update.assert_not_called()


if __name__=='__main__':unittest.main()
