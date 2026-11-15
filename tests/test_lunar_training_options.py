"""CLI configuration only; never starts a trainer."""
import os
import unittest
from unittest.mock import patch
from training_options import configure


class LunarTrainingOptions(unittest.TestCase):
    def test_detailed_physics_logs_are_isolated_with_and_without_push_flags(self):
        prefix=['launch.py','train','Wangyu-Moon-Stand-Lunar-v15']
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(configure(prefix+['--agent.max-iterations','300']),
                             prefix+['--agent.max-iterations','300'])
            self.assertEqual(os.environ['WANGYU_TRAIN_EXPERIMENT'],'wangyu_moon_stand_lunar_v15_groundsync1_detail3_no_push')
        with patch.dict(os.environ,{},clear=True):
            argv=configure(prefix+['--pushes','--push-force','200','--push-torque','50','--agent.max-iterations','300'])
            self.assertEqual(argv,prefix+['--agent.max-iterations','300'])
            self.assertEqual(os.environ['WANGYU_TRAIN_EXPERIMENT'],
                             'wangyu_moon_stand_lunar_v15_groundsync1_detail3_force200_torque50')


if __name__=='__main__':unittest.main()
