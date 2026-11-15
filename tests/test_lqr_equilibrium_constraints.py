"""Regression checks for candidate selection; no dynamics or gain preparation."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/prepare_stand_lqr.py'
spec=importlib.util.spec_from_file_location('prepare_lqr_constraints',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EquilibriumConstraints(unittest.TestCase):
    def check(self,ctrl,root=None):
        if root is None:
            root=np.array([0.,0.,1.01281276e-6,-2.20210467e-9,2.62061649e-5,1.36851058e-10])
        return module.candidate_diagnostics(root,.068,np.array(ctrl),
                    np.array([0.,-.5]),np.array([.5,0.]),207.318,['Knee','Elbow'])

    def test_small_root_residual_does_not_hide_target_failure(self):
        result=self.check([-1e-4,-.1])
        self.assertFalse(result['feasible'])
        self.assertIn('Knee target=',result['reasons'][0])
        self.assertIn('violation=',result['reasons'][0])
        self.assertAlmostEqual(result['max_target_violation_rad'],1e-4)

    def test_valid_candidate_wins_over_smaller_root_residual(self):
        invalid=self.check([-1e-4,-.1],np.zeros(6))
        valid=self.check([.01,-.1],np.array([.2,0.,0.,0.,.1,0.]))
        self.assertTrue(valid['feasible'])
        self.assertLess(valid['rank'],invalid['rank'])

    def test_only_roundoff_is_tolerated(self):
        self.assertTrue(self.check([-1e-9,1e-9])['feasible'])
        self.assertFalse(self.check([-1e-7,0.])['feasible'])

    def test_nonfinite_candidate_is_rejected(self):
        self.assertFalse(self.check([np.nan,-.1])['feasible'])

    def test_stationary_limited_closed_loop_is_accepted(self):
        # User regression: 289 clipped steps with a stationary actual root.
        # Clipping count is a diagnostic, never an instability verdict itself.
        self.assertEqual(module.rollout_motion_failures([5.01e-8,1e-7,1e-6]),[])

    def test_root_stationary_but_joints_moving_is_rejected(self):
        failures=module.rollout_motion_failures([5.01e-8,0.,.5])
        self.assertEqual(len(failures),1)
        self.assertIn('joint speed',failures[0])
        self.assertTrue(module.rollout_motion_failures([np.nan,0.,0.]))


if __name__=='__main__': unittest.main()
