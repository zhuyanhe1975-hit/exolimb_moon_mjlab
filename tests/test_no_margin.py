"""No-margin input contract and CPU config tests; no physics or training."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
import balance_mdp
import no_margin_mdp as mdp
import flat_no_margin_task as task
from standing_diagnostics import guarded_observation


class NoMargin(unittest.TestCase):
    def test_true_feature_bundle_drops_only_lipm_prediction(self):
        native={'root_quat':[1,0,0,0],'torso_quat':[1,0,0,0],
            'com_pos':[.05,0,1.],'com_vel':[.2,.1,0],
            'left_pos':[0,.1,0],'right_pos':[0,-.1,0],
            'left_load':[100.],'right_load':[100.]}
        parts=[torch.zeros(1,balance_mdp.STATE_DIM)]
        for name,_,_,_,width in balance_mdp.SENSORS:
            parts.append(torch.tensor([native.get(name,[0.]*width)],dtype=torch.float32))
        parts.append(torch.zeros(1,30))
        x=torch.cat(parts,-1)
        lunar=balance_mdp.features(x,207.318,1.635)
        earth=balance_mdp.features(x,207.318,9.81)
        self.assertFalse(torch.equal(lunar[:,95:97],earth[:,95:97]))
        raw=mdp.without_prediction(lunar);other=mdp.without_prediction(earth)
        self.assertEqual(raw.shape,(1,143));torch.testing.assert_close(raw,other)
        torch.testing.assert_close(raw[:,:95],lunar[:,:95])
        torch.testing.assert_close(raw[:,95:],lunar[:,97:145])

    def test_observation_never_calls_support_margin_bundle(self):
        state=torch.randn(2,145)
        with patch.object(mdp.base,'observation',return_value=state), \
             patch('support_mdp.data',side_effect=AssertionError('Margin bundle must not execute')):
            result=mdp.observation(object())
        self.assertEqual(result.shape,(2,143))

    def test_airborne_uses_contact_mask_without_hull(self):
        c=SimpleNamespace(found=torch.zeros(3,8),force=torch.zeros(3,8,3),
                          dist=torch.zeros(3,8),pos=torch.zeros(3,8,3))
        c.found[0,0]=c.found[2,0]=1;c.force[0,0,0]=c.force[2,0,0]=5
        c.pos[2,0,0]=float('nan')
        env=SimpleNamespace(scene={'support_contacts':SimpleNamespace(data=c)})
        with patch('support_mdp.compute_geometry',side_effect=AssertionError('No hull for airborne')):
            torch.testing.assert_close(mdp.airborne(env),torch.tensor([0.,1.,1.]))

    def test_actor_and_critic_143_without_margin_reward_or_metrics(self):
        for play in (False,True):
            cfg=task.terrain_stand_cfg(play)
            self.assertEqual(cfg.scene.num_envs,1 if play else 8192)
            self.assertEqual(cfg.episode_length_s,10.)
            self.assertNotIn('capture_risk',cfg.rewards)
            self.assertEqual(set(cfg.rewards),set(task.REWARD_WEIGHTS))
            self.assertFalse(any('margin' in name or 'support_has_area'==name or 'predicted_support' in name for name in cfg.metrics))
            for group in ('actor','critic'):
                term=cfg.observations[group].terms['state']
                self.assertIs(term.func,mdp.guarded_observation)
                probe=term.func(SimpleNamespace(num_envs=2,device='cpu'))
                self.assertEqual(probe.shape,(2,143))
            self.assertEqual(cfg.rewards['torso_velocity'].weight,-4.)
            self.assertEqual(cfg.rewards['torso_angular_velocity'].weight,-.2)
            self.assertEqual(cfg.rewards['centroidal_angular_momentum'].weight,-.01)
            self.assertEqual(cfg.rewards['joint_pose'].weight,-.25)


if __name__=='__main__':unittest.main()
