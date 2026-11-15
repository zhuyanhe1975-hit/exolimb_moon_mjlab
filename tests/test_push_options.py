"""Opt-in random force configuration; CPU only, no physics."""
import argparse
import os
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import tempfile
from pathlib import Path
import torch
from push_options import add_arguments,resolve,apply
from training_options import configure
import lunar_finetune_task as lunar
import jump_task as jump


class ExplicitPushes(unittest.TestCase):
    def args(self,argv):
        parser=argparse.ArgumentParser();add_arguments(parser);return parser.parse_args(argv)

    def test_default_off_explicit_on_and_explicit_off_wins(self):
        for argv,expected in (([],(False,0.,0.)),(['--pushes'],(True,30.,0.)),
            (['--push-force','20'],(True,20.,0.)),(['--push-torque','5'],(True,30.,5.)),
            (['--no-pushes','--push-force','20'],(False,0.,0.))):
            self.assertEqual(resolve(self.args(argv)),expected)
        for argv in (['--push-force','-1'],['--push-force','nan'],['--push-torque','inf']):
            with self.assertRaises(ValueError):resolve(self.args(argv))

    def test_stand_and_hop_configs_off_by_default(self):
        with patch.dict(os.environ,{},clear=True):
            for task in (lunar,jump):
                for play in (False,True):self.assertNotIn('pelvis_push',task.terrain_stand_cfg(play).events)

    def test_explicit_enable_adds_native_event_even_when_base_is_off(self):
        with patch.dict(os.environ,{},clear=True):
            cfg=jump.terrain_stand_cfg(True)
            original_weights={k:v.weight for k,v in cfg.rewards.items()}
            apply(cfg,True,20.,5.)
            self.assertEqual(cfg.events['pelvis_push'].params['force_range'],(-20.,20.))
            self.assertEqual(cfg.events['pelvis_push'].params['torque_range'],(-5.,5.))
            apply(cfg,False)
            self.assertNotIn('pelvis_push',cfg.events)
            self.assertEqual(original_weights,{k:v.weight for k,v in cfg.rewards.items()})

    def test_training_without_flags_overrides_stale_enabled_environment(self):
        with patch.dict(os.environ,{'WANGYU_TRAIN_PUSH_ENABLED':'1','WANGYU_TRAIN_PUSH_FORCE':'200'}):
            configure(['launch.py','train',lunar.TASK_ID,'--agent.max-iterations','100'])
            self.assertEqual(os.environ['WANGYU_TRAIN_PUSH_ENABLED'],'0')
            self.assertTrue(os.environ['WANGYU_TRAIN_EXPERIMENT'].endswith('_no_push'))
            self.assertNotIn('pelvis_push',lunar.terrain_stand_cfg().events)

    def test_explicit_training_parameters_enable_requested_force(self):
        with patch.dict(os.environ,{},clear=True):
            configure(['launch.py','train',lunar.TASK_ID,'--pushes','--push-force','20'])
            self.assertEqual(os.environ['WANGYU_TRAIN_PUSH_ENABLED'],'1')
            self.assertEqual(lunar.terrain_stand_cfg().events['pelvis_push'].params['force_range'],(-20.,20.))

    def test_hop_training_cli_parses_opt_in_and_keeps_log_isolation(self):
        from jump_options import configure as configure_hop,TASK_ID
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'model_299.pt'
            torch.save({'actor_state_dict':{'mlp.0.weight':torch.zeros(1,146),
                                          'mlp.4.weight':torch.zeros(23,1)}},source)
            for flags,on in (([],False),(['--pushes','--push-force','20'],True),(['--no-pushes'],False)):
                with patch.dict(os.environ,{},clear=True):
                    argv=configure_hop(['launch.py','train',TASK_ID,'--standing-prior',str(source),*flags])
                    self.assertEqual(argv,['launch.py','train',TASK_ID])
                    self.assertEqual(os.environ['WANGYU_TRAIN_PUSH_ENABLED'],str(int(on)))
                    self.assertTrue(os.environ['WANGYU_JUMP_EXPERIMENT'].endswith('_force20_torque0' if on else '_no_push'))
                    self.assertEqual('pelvis_push' in jump.terrain_stand_cfg().events,on)


if __name__=='__main__':unittest.main()
