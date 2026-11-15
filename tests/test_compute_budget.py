"""Policy-independent MJWarp capacity bound for lunar v15; CPU static forward only, no time stepping."""
from types import SimpleNamespace
import itertools
import os
import unittest
import warnings
from unittest.mock import patch
import mujoco
import torch
from mjlab.scene import Scene
import lunar_finetune_task as task
import standing_diagnostics as diag

# MJWarp's hfield collider keeps the deepest contact plus at most three more
# (collision_convex.ccd_hfield_kernel: "contact 1/2/3"), i.e. <=4 per pair.
HFIELD_CONTACTS_PER_PAIR = 4


def candidate_pairs(model):
    colliding = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    excluded = {int(s) for s in model.exclude_signature}
    pairs = []
    for i, j in itertools.combinations(colliding, 2):
        if not ((model.geom_contype[i] & model.geom_conaffinity[j]) or
                (model.geom_contype[j] & model.geom_conaffinity[i])):
            continue
        bi, bj = int(model.geom_bodyid[i]), int(model.geom_bodyid[j])
        if bi == bj or ((min(bi, bj) << 16) + max(bi, bj)) in excluded:
            continue
        pairs.append((i, j))
    return pairs


class ComputeBudget(unittest.TestCase):
    def compile(self, play):
        cfg = task.terrain_stand_cfg(play=play)
        cfg.scene.num_envs = 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return cfg, Scene(cfg.scene, 'cpu').compile()

    def test_capacities_cover_worst_case_from_compiled_model(self):
        for play in (False, True):
            cfg, model = self.compile(play)
            # Mirror Simulation._init_with_model before mjwarp.put_data.
            # CPU hfield collision is not bounded by the GPU four-contact rule.
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            self.assertLessEqual(data.ncon, cfg.sim.nconmax, f'CPU upload play={play}')
            self.assertLessEqual(data.nefc, cfg.sim.njmax, f'CPU upload play={play}')
            self.assertEqual(data.time, 0.)
            pairs = candidate_pairs(model)
            hfield = int(mujoco.mjtGeom.mjGEOM_HFIELD)
            box = int(mujoco.mjtGeom.mjGEOM_BOX)
            # Every pair must be box-vs-hfield, or the 4-contact bound is invalid.
            for i, j in pairs:
                self.assertEqual(sorted((int(model.geom_type[i]), int(model.geom_type[j]))),
                                 sorted((box, hfield)))
            self.assertEqual(len(pairs), 2)
            contacts = HFIELD_CONTACTS_PER_PAIR * len(pairs)
            self.assertEqual(int(model.opt.cone), int(mujoco.mjtCone.mjCONE_PYRAMIDAL))
            condim = max(int(model.geom_condim[g]) for pair in pairs for g in pair)
            rows_per_contact = 1 if condim == 1 else 2 * (condim - 1)
            limited = int(model.jnt_limited.sum())
            # A hinge whose range is wider than twice its margin can activate at
            # most one side at a time.
            width = model.jnt_range[:, 1] - model.jnt_range[:, 0]
            self.assertTrue(((width > 2 * model.jnt_margin) | (model.jnt_limited == 0)).all())
            self.assertEqual((model.neq, model.ntendon, int((model.dof_frictionloss > 0).sum())), (0, 0, 0))
            nefc = contacts * rows_per_contact + limited
            self.assertLessEqual(contacts, cfg.sim.nconmax, f'play={play}')
            self.assertLess(nefc, cfg.sim.njmax, f'play={play}')
            # One primary sole against the terrain per contact sensor.
            self.assertLessEqual(HFIELD_CONTACTS_PER_PAIR, cfg.sim.contact_sensor_maxmatch)
        self.assertEqual((cfg.sim.nconmax, cfg.sim.njmax, cfg.sim.contact_sensor_maxmatch), (128, 512, 8))

    def test_measured_peak_is_within_bound(self):
        """The GPU audit the user ran (nefc peak 31) must stay under the derived bound."""
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'outputs/stand_lunar_v15/compute_budget.json'
        if not path.exists():
            self.skipTest('no measured audit report present')
        report = json.loads(path.read_text())
        self.assertEqual(report['overflow_bitmask'], 0)
        self.assertLess(report['measured_peak']['nefc'], 64)
        self.assertLessEqual(report['measured_peak']['nacon'], 8 * report['num_envs'])


class OverflowGuard(unittest.TestCase):
    def env(self, overflow, step):
        return SimpleNamespace(num_envs=overflow.numel(), device='cpu', common_step_counter=step,
                               sim=SimpleNamespace(data=SimpleNamespace(overflow=overflow)),
                               cfg=SimpleNamespace(sim=SimpleNamespace(nconmax=16, njmax=64,
                                                                       contact_sensor_maxmatch=8)))

    def test_overflow_is_accumulated_across_resets_and_raised_at_rollout_boundary(self):
        overflow = torch.zeros(4, dtype=torch.int32)
        env = self.env(overflow, 1)
        with patch.dict(os.environ, {'WANGYU_OVERFLOW_CHECK_STEPS': '4'}):
            overflow[2] = 1  # NEFC on a world that MJWarp will clear on reset.
            diag.overflow_guard(env)
            overflow.zero_()  # reset_data cleared it; the accumulator must not.
            env.common_step_counter = 2
            diag.overflow_guard(env)
            env.common_step_counter = 4
            with self.assertRaisesRegex(RuntimeError, r"\['NEFC'\] in 1 worlds"):
                diag.overflow_guard(env)

    def test_clean_run_does_not_raise_or_sync_every_step(self):
        overflow = torch.zeros(3, dtype=torch.int32)
        env = self.env(overflow, 1)
        for step in range(1, 70):
            env.common_step_counter = step
            diag.overflow_guard(env)
        self.assertFalse(env._standing_overflow.any())

    def test_cpu_fakes_without_overflow_field_are_ignored(self):
        env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace()))
        self.assertIsNone(diag.overflow_guard(env))


if __name__ == '__main__':
    unittest.main()
