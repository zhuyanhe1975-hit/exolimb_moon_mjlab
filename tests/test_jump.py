"""Single reward regression: original lunar standing configuration is preserved."""
from dataclasses import fields,is_dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import math
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from mjlab.scene import Scene
import jump_task as task
import jump_mdp as mdp
import lunar_finetune_task as base
from jump_options import select_prior


def config_signature(value):
    # MJLab creates fresh spec wrapper functions on each config build. Compare
    # their implementation identity, rather than distinct Python addresses.
    if is_dataclass(value):return {f.name:config_signature(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,dict):return {k:config_signature(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [config_signature(v) for v in value]
    if callable(value):return (value.__module__,value.__qualname__,getattr(getattr(value,'__code__',None),'co_code',None))
    return value


class UpwardMomentumOnly(unittest.TestCase):
    def env(self,velocity,gravity=(0.,0.,-1.635)):
        # Two unrelated sensor columns in front ensure address lookup is used.
        values=torch.cat((torch.full((len(velocity),2),99.),torch.tensor(velocity)),1)
        model=SimpleNamespace(sensor=lambda name:SimpleNamespace(dim=[3],adr=[2]),
                              opt=SimpleNamespace(gravity=np.array(gravity)),body=lambda name:SimpleNamespace(id=0),
                              body_subtreemass=np.array([100.]))
        return SimpleNamespace(sim=SimpleNamespace(mj_model=model,data=SimpleNamespace(sensordata=values)),
            common_step_counter=0,step_dt=1/60,num_envs=len(velocity),device='cpu',reset_buf=torch.zeros(len(velocity),dtype=torch.bool))

    def test_rest_descent_and_horizontal_motion_are_zero(self):
        e=self.env([[0.,0.,0.],[0.,0.,-.4],[.7,-.8,0.],[0.,0.,.3],[0.,0.,2.]])
        target=100.*np.sqrt(2*1.635*.25)
        torch.testing.assert_close(mdp.upward_momentum(e),torch.tensor([0.,-40.,0.,30.,200.]))
        torch.testing.assert_close((mdp.upward_momentum(e)/mdp.target_momentum(e,.25)).clamp(0,1),torch.tensor([0.,0.,0.,30./target,1.],dtype=torch.float32))
        torch.testing.assert_close(mdp.upward_speed(e),torch.tensor([0.,-.4,0.,.3,2.]))

    def test_direction_follows_actual_gravity_not_z(self):
        e=self.env([[-.4,0.,0.],[.4,0.,0.],[0.,0.,.7]],gravity=(1.635,0.,0.))
        torch.testing.assert_close(mdp.upward_momentum(e),torch.tensor([40.,-40.,0.]))
        # Direction changes invalidate the cached direction tensor.
        e.sim.mj_model.opt.gravity[:]=[0.,0.,1.635]
        torch.testing.assert_close(mdp.upward_momentum(e),torch.tensor([0.,0.,-70.]))
        e=self.env([[0.,.3,.4]],gravity=(0.,-3.,-4.))
        torch.testing.assert_close(mdp.upward_momentum(e),torch.tensor([50.]))
        np.testing.assert_allclose(mdp.anti_gravity_direction((0.,-30.,-40.)),[0.,.6,.8])

    def test_invalid_gravity_fails_clearly(self):
        for g in ((0.,0.,0.),(float('nan'),0.,-1.),(0.,float('inf'),0.)):
            with self.assertRaises(ValueError):mdp.anti_gravity_direction(g)

    def test_height_and_native_mass_calibrate_target(self):
        e=self.env([[0.,0.,.3],[0.,0.,.3]])
        e.sim.model=SimpleNamespace(body_subtreemass=torch.tensor([[100.],[200.]]))
        target=mdp.target_momentum(e,.25)
        torch.testing.assert_close(target,torch.tensor([100.,200.])*np.sqrt(2*1.635*.25))
        torch.testing.assert_close(mdp.upward_momentum(e),torch.tensor([30.,60.]))
        reward=(mdp.upward_momentum(e)/mdp.target_momentum(e,.25)).clamp(0,1)
        self.assertAlmostEqual(float(reward[0]),float(reward[1]),places=6)
        torch.testing.assert_close(mdp.target_momentum(e,1.),2*target)
        e.sim.model.body_subtreemass*=1.5 # Read backend mass, not stale CPU template.
        torch.testing.assert_close(mdp.target_momentum(e,.25),1.5*target)
        for h in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):mdp.target_momentum(e,h)

    def test_only_one_added_reward_every_original_setting_kept(self):
        for play in (False,True):
            cfg=task.terrain_stand_cfg(play);original=base.terrain_stand_cfg(play)
            self.assertEqual(set(cfg.rewards)-set(original.rewards),{'upward_momentum','jump_height'})
            self.assertEqual(set(original.rewards)-set(cfg.rewards),set())
            for name,term in original.rewards.items():self.assertEqual(cfg.rewards[name],term)
            self.assertEqual(cfg.rewards['upward_momentum'].weight,task.REWARD_WEIGHTS['upward_momentum'])
            # All physics/observation/reset/control/episode/disturbance config unchanged.
            for field in fields(cfg):
                if field.name not in ('rewards','metrics','events'):
                    self.assertEqual(config_signature(getattr(cfg,field.name)),config_signature(getattr(original,field.name)),field.name)
            self.assertEqual(set(cfg.events),set(original.events))
            self.assertEqual(cfg.events['reset_pose'].params,original.events['reset_pose'].params)
            self.assertEqual(set(cfg.events['reset_pose'].func.model_fields),set(original.events['reset_pose'].func.model_fields))
            self.assertEqual(cfg.rewards['jump_height'].weight,10.)
            self.assertEqual(set(cfg.metrics)-set(original.metrics),{'com_upward_speed_m_s','com_upward_momentum_kg_m_s','target_upward_momentum_kg_m_s','jump_com_rise_m','jump_com_rise_peak_m','jump_airborne_com_rise_peak_m','jump_height_airborne_fraction'})
            self.assertEqual(task.OBS_DIM,146)
            self.assertEqual(cfg.episode_length_s,10.)
            for group in ('actor','critic'):
                self.assertEqual(cfg.observations[group].terms['state'].func(
                    SimpleNamespace(num_envs=2,device='cpu')).shape,(2,146))
        self.assertTrue(task.EXPERIMENT.startswith('wangyu_moon_hop_lunar_v2_'))

    def test_native_subtree_sensor_and_projection(self):
        cfg=task.terrain_stand_cfg();cfg.scene.num_envs=1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');m=Scene(cfg.scene,'cpu').compile()
        d=mujoco.MjData(m);d.qvel[:3]=[.1,.2,.3]
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_comVel(m,d)
        mujoco.mj_subtreeVel(m,d);mujoco.mj_sensorPos(m,d);mujoco.mj_sensorVel(m,d)
        s=m.sensor('robot/balance_com_vel')
        self.assertEqual(int(s.type[0]),int(mujoco.mjtSensor.mjSENS_SUBTREELINVEL))
        self.assertEqual(int(s.objid[0]),m.body('robot/Pelvis').id)
        env=SimpleNamespace(sim=SimpleNamespace(mj_model=m,data=SimpleNamespace(sensordata=torch.tensor(d.sensordata)[None])))
        expected=np.dot(d.subtree_linvel[m.body('robot/Pelvis').id],mdp.anti_gravity_direction(m.opt.gravity))
        self.assertAlmostEqual(float(mdp.upward_speed(env)[0]),expected,places=12)
        self.assertAlmostEqual(float(mdp.upward_momentum(env)[0]),float(m.body_subtreemass[m.body('robot/Pelvis').id])*expected,places=10)
        pos_sensor=m.sensor('robot/balance_com_pos');address=int(pos_sensor.adr[0])
        self.assertEqual(int(pos_sensor.type[0]),int(mujoco.mjtSensor.mjSENS_SUBTREECOM))
        np.testing.assert_allclose(d.sensordata[address:address+3],d.subtree_com[m.body('robot/Pelvis').id],atol=1e-12)
        env.common_step_counter=0;env.num_envs=1;env.device='cpu';env.reset_buf=torch.zeros(1,dtype=torch.bool)
        found=torch.ones(1,8)
        contact=SimpleNamespace(found=found,dist=torch.zeros(1,8),pos=torch.zeros(1,8,3))
        with patch('support_mdp.phase_contact_data',return_value=contact):
            self.assertEqual(float(mdp.height_reward(env,.25)[0]),0.)
            # Translate the actual native articulated model upward, no mj_step.
            d.qpos[2]+=.25
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_sensorPos(m,d)
            env.sim.data.sensordata=torch.tensor(d.sensordata)[None]
            env.common_step_counter+=1;found.zero_()
            self.assertAlmostEqual(float(mdp.measured_com_rise(env)[0]),.25,places=12)
            self.assertAlmostEqual(float(mdp.height_reward(env,.25)[0]),1.,places=12)
        self.assertEqual(d.time,0.)

    def test_dense_instantaneous_reward_has_no_contact_or_history_requirement(self):
        e=self.env([[0.,0.,0.]])
        v=mdp.target_upward_speed(e,.25)
        # Reading contacts would be a regression: upward motion while supported
        # must receive the same reward as the identical velocity in flight.
        with patch('support_mdp.phase_contact_data',side_effect=AssertionError('unexpected contact gate')):
            for velocity,score in ((0.,0.),(.5*v,.5),(v,1.),(-v,0.),(.25*v,.25),(2*v,1.)):
                e.sim.data.sensordata[:,4]=velocity
                expected=math.expm1(5*score)/math.expm1(5)
                self.assertAlmostEqual(float(mdp.momentum_reward(e,.25)[0]),expected,places=6)
                self.assertAlmostEqual(float(mdp.momentum_reward(e,.25)[0]),expected,places=6)
        self.assertFalse(hasattr(e,'_jump_previous_support'))
        self.assertFalse(hasattr(e,'_jump_takeoff_values'))

    def test_exponential_shape_suppresses_many_small_momentum_rewards(self):
        x=torch.tensor([-2.,0.,.1,.2,.5,.8,1.,2.],dtype=torch.float64)
        y=mdp.exponential_momentum_fraction(x,5.)
        expected=torch.tensor([0.,0.,.00440070123080884,.011656230956039607,
                               .07585818002124355,.3635913534411692,1.,1.],dtype=torch.float64)
        torch.testing.assert_close(y,expected,atol=1e-12,rtol=1e-12)
        self.assertTrue((y[1:]>=y[:-1]).all())
        # Equal duration: ten 10%-target samples no longer match a full-target
        # sample as they did linearly. This is not a no-cumulative-return claim.
        self.assertLess(float(10*y[2]),float(y[6]))
        self.assertLess(float(2*y[4]),float(y[6]))
        self.assertTrue(torch.isfinite(mdp.exponential_momentum_fraction(x.float(),50.)).all())
        for k in (0.,-1.,float('nan'),float('inf'),51.):
            with self.assertRaises(ValueError):mdp.exponential_momentum_fraction(x,k)
        cfg=task.terrain_stand_cfg()
        self.assertEqual(cfg.rewards['upward_momentum'].params['exponent'],5.)
        self.assertIn('momentum_height_exp1_km5_kh5_',task.EXPERIMENT)

    def test_prior_auto_selection_excludes_old_invalid_terrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=root/'wangyu_moon_stand_lunar_v15'/'2099-01-01';old.mkdir(parents=True);(old/'model_799.pt').touch()
            flat=root/'wangyu_moon_stand_flat_v14'/'2026-01-01';flat.mkdir(parents=True);(flat/'model_500.pt').touch()
            self.assertEqual(select_prior(root),flat/'model_500.pt')
            fixed=root/'wangyu_moon_stand_lunar_v15_groundsync1_detail3'/'2026-02-01';fixed.mkdir(parents=True);(fixed/'model_299.pt').touch()
            self.assertEqual(select_prior(root),fixed/'model_299.pt')


class JumpHeightReward(unittest.TestCase):
    def setUp(self):
        n=2
        self.com=torch.tensor([[3.,-4.,17.],[0.,0.,2.]])
        self.found=torch.zeros(n,8)
        self.contact=SimpleNamespace(found=self.found,dist=torch.zeros(n,8),pos=torch.zeros(n,8,3))
        self.env=SimpleNamespace(common_step_counter=0,num_envs=n,device='cpu',reset_buf=torch.zeros(n,dtype=torch.bool),
            sim=SimpleNamespace(mj_model=SimpleNamespace(sensor=lambda name:SimpleNamespace(dim=[3],adr=[0]),
                opt=SimpleNamespace(gravity=np.array([0.,0.,-1.635]))),data=SimpleNamespace(sensordata=self.com)))

    def step(self):
        self.env.common_step_counter+=1
        return mdp.height_reward(self.env,.25,5.)

    def test_spawn_gap_stance_extension_and_one_foot_are_not_height_rewards(self):
        with patch('support_mdp.phase_contact_data',return_value=self.contact):
            self.assertTrue((self.step()==0).all()) # spawn air has no baseline
            self.found[:]=1
            self.assertTrue((self.step()==0).all()) # first contact records native COM
            self.com[:,2]+=.25
            self.assertTrue((self.step()==0).all()) # supported extension
            self.found.zero_();self.found[:,4:]=1
            self.assertTrue((self.step()==0).all()) # last foot remains supported
            self.found.zero_()
            torch.testing.assert_close(self.step(),torch.ones(2))
            # Falling through the same height can earn height reward; it is not
            # incorrectly gated by upward velocity or the momentum reward.
            self.com[:,2]-=.125
            torch.testing.assert_close(self.step(),torch.full((2,),.0758581800),atol=1e-6,rtol=1e-6)

    def test_reference_is_per_world_and_partial_reset_is_independent(self):
        with patch('support_mdp.phase_contact_data',return_value=self.contact):
            self.found[:]=1;self.step()
            self.com[:,2]+=.25;self.found.zero_();self.step()
            with patch.object(task.base,'reset_local'):
                task.reset_jump(self.env,torch.tensor([0]))
            result=self.step()
            torch.testing.assert_close(result,torch.tensor([0.,1.]))
            # New world-0 altitude becomes its own reference on contact, rather
            # than inheriting world-1 or a previous terrain/reset altitude.
            self.com[0,2]=100.;self.found[0]=1;self.step()
            self.com[0,2]=100.25;self.found[0]=0
            torch.testing.assert_close(self.step(),torch.ones(2))

    def test_height_uses_anti_gravity_and_failure_does_not_reward(self):
        self.env.sim.mj_model.opt.gravity[:]=[1.635,0.,0.]
        with patch('support_mdp.phase_contact_data',return_value=self.contact):
            self.found[:]=1;self.step()
            self.found.zero_();self.com[:,0]-=.25
            torch.testing.assert_close(self.step(),torch.ones(2))
            self.env.reset_buf[0]=True
            torch.testing.assert_close(self.step(),torch.tensor([0.,1.]))
            self.env.reset_buf[:]=False
            self.com[:,0]+=.25;self.com[:,2]+=2.
            self.assertTrue((self.step()==0).all()) # world Z isn't up for this g

    def test_reward_and_metric_reads_share_one_baseline_snapshot(self):
        with patch('support_mdp.phase_contact_data',return_value=self.contact):
            self.found[:]=1;self.step()
            before=self.env._jump_height_baseline.clone()
            mdp.measured_com_rise(self.env);mdp.height_airborne_fraction(self.env)
            mdp.height_reward(self.env,.25,5.)
            torch.testing.assert_close(before,self.env._jump_height_baseline,rtol=0,atol=0)



if __name__=='__main__':unittest.main()
