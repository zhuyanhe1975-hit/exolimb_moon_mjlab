"""Native event lifecycle on CPU fake buffers; no environment or physics run."""
from types import SimpleNamespace
import unittest
import os
from unittest.mock import patch

import torch
from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from stand_push_task import HorizontalBodyImpulse,PUSH_PARAMS,terrain_stand_cfg,push_force


class Asset:
    num_bodies=13

    def __init__(self,n):
        self.data=SimpleNamespace(body_external_wrench=torch.zeros((n,1,6)))

    def write_external_wrench_to_sim(self,forces,torques,env_ids=None,body_ids=None):
        assert body_ids==[0]
        ids=slice(None) if env_ids is None else env_ids
        self.data.body_external_wrench[ids,:,:3]=forces
        self.data.body_external_wrench[ids,:,3:]=torques


class PelvisPush(unittest.TestCase):
    def setup_event(self):
        torch.manual_seed(42)
        asset=Asset(32)
        env=SimpleNamespace(scene={'robot':asset},num_envs=32,device='cpu',step_dt=1/60)
        selection=SceneEntityCfg('robot',body_names=('Pelvis',));selection.body_ids=[0]
        params={**PUSH_PARAMS,'asset_cfg':selection}
        cfg=EventTermCfg(func=HorizontalBodyImpulse,mode='step',params=params)
        return env,asset,HorizontalBodyImpulse(cfg,env),params

    def test_native_term_and_initial_cooldown(self):
        env,asset,event,params=self.setup_event()
        self.assertIsInstance(event,apply_body_impulse)
        self.assertTrue((event._interval_time_left>=2).all())
        self.assertTrue((event._interval_time_left<=4).all())
        for _ in range(30):event(env,None,**params)
        self.assertFalse(asset.data.body_external_wrench.any())

    def test_force_is_horizontal_bounded_and_has_no_extra_torque(self):
        env,asset,event,params=self.setup_event()
        event._interval_time_left.zero_();event(env,None,**params)
        wrench=asset.data.body_external_wrench
        self.assertTrue((wrench[:,:,:2].abs()<=30).all())
        self.assertTrue(wrench[:,:,:2].any())
        self.assertFalse(wrench[:,:,2:].any())
        self.assertTrue(event._active.all())
        self.assertTrue((event._time_remaining>=.1).all())
        self.assertTrue((event._time_remaining<=.2).all())
        self.assertTrue((push_force(env,params['asset_cfg'])<=30*2**.5).all())

    def test_random_torque_is_bounded_held_and_cleared(self):
        env,asset,event,params=self.setup_event()
        params['torque_range']=(-10.,10.)
        event._interval_time_left.zero_();event(env,None,**params)
        torque=asset.data.body_external_wrench[:,:,3:].clone()
        self.assertTrue((torque.abs()<=10).all())
        self.assertTrue((torque>0).any() and (torque<0).any())
        event(env,None,**params)
        torch.testing.assert_close(asset.data.body_external_wrench[:,:,3:],torque)
        event.reset()
        self.assertFalse(asset.data.body_external_wrench.any())

    def test_partial_reset_and_expiry_clear_only_intended_wrenches(self):
        env,asset,event,params=self.setup_event()
        event._interval_time_left.zero_();event(env,None,**params)
        old=asset.data.body_external_wrench.clone()
        ids=torch.tensor([2,7]);event.reset(ids)
        self.assertFalse(asset.data.body_external_wrench[ids].any())
        torch.testing.assert_close(asset.data.body_external_wrench[0],old[0])
        self.assertFalse(event._active[ids].any())
        self.assertTrue((event._interval_time_left[ids]>=2).all())
        event._time_remaining.zero_();event(env,None,**params)
        self.assertFalse(event._active.any())
        self.assertFalse(asset.data.body_external_wrench.any())

    def test_train_and_play_keep_same_perturbation_and_scene(self):
        for play in (False,True):
            with patch.dict(os.environ,{'WANGYU_TRAIN_PUSH_ENABLED':'1','WANGYU_TRAIN_PUSH_FORCE':'30','WANGYU_TRAIN_PUSH_TORQUE':'0'}):
                cfg=terrain_stand_cfg(play)
            term=cfg.events['pelvis_push']
            self.assertEqual(term.mode,'step')
            self.assertEqual(term.params['asset_cfg'].body_names,('Pelvis',))
            self.assertEqual(cfg.scene.num_envs,1 if play else 8192)
            self.assertEqual(cfg.scene.terrain.terrain_type,'generator')
            self.assertEqual(cfg.actions['position'].scale,.25)
            self.assertNotIn('station_drift',cfg.rewards)
            self.assertIn('pelvis_push_active_fraction',cfg.metrics)


if __name__=='__main__':unittest.main()
