"""CPU fake-state guards only; no physics or policy execution."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
import standing_diagnostics as diag


class StandingDiagnostics(unittest.TestCase):
    def state(self,n=3):
        q=torch.zeros(n,30);q[:,3]=1
        v=torch.zeros(n,29);limits=torch.tensor([[-.5,.5]]*23)
        return q,v,limits

    def test_finite_joint_explosion_and_native_nonfinite(self):
        q,v,limits=self.state()
        q[1,8]=1000.;v[2,4]=float('nan')
        native_nonfinite=~torch.isfinite(v).all(-1)
        reasons=diag.anomaly_reasons(q,v,limits,native_nonfinite)
        self.assertEqual(reasons.any(-1).tolist(),[False,True,True])
        self.assertTrue(reasons[1,1]);self.assertTrue(reasons[2,0])

    def test_soft_constraint_excursion_not_treated_as_explosion(self):
        q,v,limits=self.state()
        q[:,7:]=.6;v[:,6:]=10.
        self.assertFalse(diag.anomaly_reasons(q,v,limits,torch.zeros(3,dtype=torch.bool)).any())

    def test_quaternion_and_gross_speed(self):
        q,v,limits=self.state();q[0,3:7]=0;v[1,0]=51;v[2,7]=201
        reasons=diag.anomaly_reasons(q,v,limits,torch.zeros(3,dtype=torch.bool))
        self.assertTrue(reasons[0,4]);self.assertTrue(reasons[1,2]);self.assertTrue(reasons[2,3])

    def test_gross_native_acceleration_is_detected(self):
        q,v,limits=self.state();a=torch.zeros_like(v);a[1,2]=1e10
        reasons=diag.anomaly_reasons(q,v,limits,torch.zeros(3,dtype=torch.bool),a)
        self.assertTrue(reasons[1,5]);self.assertFalse(reasons[0].any())

    def test_guard_raises_before_reward_can_be_evaluated(self):
        q,v,limits=self.state();q[1,7]=1e6
        native=SimpleNamespace(qpos=q,qvel=v,qacc=torch.zeros_like(v),
            qacc_warmstart=torch.zeros_like(v),sensordata=torch.zeros(3,4))
        env=SimpleNamespace(sim=SimpleNamespace(data=native),common_step_counter=1,
                            _standing_native_limits=limits)
        with patch.object(diag,'dump_and_raise',side_effect=RuntimeError('raw dump saved')) as dump:
            with self.assertRaisesRegex(RuntimeError,'raw dump saved'):diag.numerical_guard(env)
            self.assertTrue(dump.call_args.args[1][1].any())
            self.assertEqual(dump.call_args.args[2],'pre_reward')

    def test_unforced_metric_waits_after_birth_force_and_torque(self):
        wrench=torch.zeros(3,2,6)
        robot=SimpleNamespace(data=SimpleNamespace(body_external_wrench=wrench))
        env=SimpleNamespace(scene={'robot':robot},sim=SimpleNamespace(data=SimpleNamespace(qfrc_applied=torch.zeros(3,29))),num_envs=3,common_step_counter=0,
                            device='cpu',episode_length_buf=torch.zeros(3),step_dt=.1)
        self.assertFalse(diag.settled_mask(env).any())
        env.common_step_counter=21;env.episode_length_buf.fill_(21)
        wrench[1,0,0]=20;wrench[2,0,5]=5
        self.assertEqual(diag.settled_mask(env).tolist(),[True,False,False])
        wrench.zero_();env.common_step_counter=22
        self.assertEqual(diag.settled_mask(env).tolist(),[True,False,False])
        env.common_step_counter=41
        self.assertTrue(diag.settled_mask(env).all())

    def test_manager_shape_probe_does_not_validate_unreset_terrain_state(self):
        env=SimpleNamespace(num_envs=4,device='cpu')
        with patch.object(diag,'check_state') as check:
            obs=diag.guarded_observation(env)
            self.assertEqual(obs.shape,(4,151))
            check.assert_not_called()

    def test_real_observation_still_fails_on_nonfinite_output(self):
        env=SimpleNamespace(num_envs=2,device='cpu',common_step_counter=1,observation_manager=object())
        with patch.object(diag,'check_state',return_value=None) as check, \
             patch('support_mdp.observation',return_value=torch.full((2,151),float('nan'))), \
             patch.object(diag,'dump_and_raise',side_effect=RuntimeError('invalid observation')) as dump:
            with self.assertRaisesRegex(RuntimeError,'invalid observation'):diag.guarded_observation(env)
            # State and observation bits now share one host reduction; the state
            # mask is still built at the same post_forward boundary.
            check.assert_called_once_with(env,'post_forward',defer=True)
            self.assertTrue(dump.call_args.args[1][:,-1].all())
            self.assertEqual(dump.call_args.args[2],'observation')

    def test_fused_guard_reports_physics_stage_for_native_anomaly(self):
        """A native explosion must not be relabelled as an observation fault."""
        q,v,limits=self.state(2);q[1,7]=1e6
        native=SimpleNamespace(qpos=q,qvel=v,qacc=torch.zeros_like(v),
            qacc_warmstart=torch.zeros_like(v),sensordata=torch.zeros(2,4))
        env=SimpleNamespace(num_envs=2,device='cpu',common_step_counter=1,
            observation_manager=object(),sim=SimpleNamespace(data=native),
            _standing_native_limits=limits)
        with patch('support_mdp.observation',return_value=torch.zeros(2,151)), \
             patch.object(diag,'dump_and_raise',side_effect=RuntimeError('raw dump saved')) as dump:
            with self.assertRaisesRegex(RuntimeError,'raw dump saved'):diag.guarded_observation(env)
            self.assertEqual(dump.call_args.args[2],'post_forward')
            self.assertTrue(dump.call_args.args[1][1,1])
            self.assertFalse(dump.call_args.args[1][:,-1].any())

    def test_fused_guard_passes_clean_state_and_caches(self):
        q,v,limits=self.state(2)
        native=SimpleNamespace(qpos=q,qvel=v,qacc=torch.zeros_like(v),
            qacc_warmstart=torch.zeros_like(v),sensordata=torch.zeros(2,4))
        env=SimpleNamespace(num_envs=2,device='cpu',common_step_counter=1,
            observation_manager=object(),sim=SimpleNamespace(data=native),
            _standing_native_limits=limits)
        expected=torch.zeros(2,151)
        with patch('support_mdp.observation',return_value=expected) as observe:
            self.assertTrue(torch.equal(diag.guarded_observation(env),expected))
            self.assertTrue(torch.equal(diag.guarded_observation(env),expected))
            observe.assert_called_once()
        # The state guard key is published, so a later same-step reward-stage
        # check does not rebuild the mask.
        self.assertEqual(env._standing_guard_key,(1,0,'post_forward'))

    def test_configuration_keeps_rewards_and_guard_first(self):
        import support_task
        cfg=support_task.terrain_stand_cfg()
        self.assertEqual(next(iter(cfg.terminations)),'numerical_guard')
        self.assertIs(cfg.observations['actor'].terms['state'].func,diag.guarded_observation)
        self.assertEqual(cfg.rewards['joint_pose'].weight,-.25)
        self.assertEqual(cfg.rewards['centroidal_angular_momentum'].weight,-.002)
        self.assertIn('moving_without_wrench_fraction',cfg.metrics)


if __name__=='__main__':unittest.main()
