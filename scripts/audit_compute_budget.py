"""Measure native MJWarp capacity usage so nconmax/njmax/maxmatch are sized, not guessed.

Reads MJWarp's own per-world counters (`nefc`, `nacon`, `solver_niter`) and the
`overflow` bitmask after every control step, as the MJWarp performance
documentation prescribes: memory and compute scale with these capacities, so they
should be as small as the measured peak allows.

This script DOES advance GPU physics. It does not train, does not write
checkpoints, and does not change any configured capacity. Run it only when the
GPU is otherwise idle; concurrent training makes the timings meaningless.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

TASKS = {
    'lunar-v15': ('lunar_finetune_task', 'stand_lunar_v15'),
    'flat-v14': ('flat_torso_observation_task', 'stand_flat_v14'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=sorted(TASKS), default='lunar-v15')
    parser.add_argument('--num-envs', type=int, default=8192)
    parser.add_argument('--steps', type=int, default=64, help='control steps measured after warmup')
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--nconmax', type=int, default=None,
                        help='override the task capacity for a matched A/B; the task default is unchanged on disk')
    parser.add_argument('--njmax', type=int, default=None, help='same, per-world constraint rows')
    parser.add_argument('--contact-sensor-maxmatch', type=int, default=None)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.num_envs < 1:
        parser.error('steps/num-envs must be positive and warmup nonnegative')
    for name in ('nconmax', 'njmax', 'contact_sensor_maxmatch'):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f'{name} must be positive')

    import numpy as np
    import torch
    # The installed MJWarp keeps OverflowType in _src.types (not re-exported).
    from mujoco_warp._src.types import OverflowType
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    module_name, subdir = TASKS[args.task]
    module = __import__(module_name)
    torch.set_num_threads(1)
    configure_torch_backends()

    cfg = module.terrain_stand_cfg()
    cfg.scene.num_envs = args.num_envs
    overrides = {}
    for name in ('nconmax', 'njmax', 'contact_sensor_maxmatch'):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
            setattr(cfg.sim, name, value)
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        data = env.sim.wp_data
        action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        peaks = {'nefc': 0, 'nacon': 0, 'solver_niter': 0}
        overflow = 0
        samples = []
        with torch.inference_mode():
            env.reset(seed=42)
            for _ in range(args.warmup):
                env.step(action)
            for _ in range(args.steps):
                env.step(action)
                # One host read per control step: this is a measurement tool, not
                # the training path, where such syncs are deliberately avoided.
                nefc = int(data.nefc.numpy().max())
                nacon = int(data.nacon.numpy()[0])
                niter = int(data.solver_niter.numpy().max())
                overflow |= int(np.bitwise_or.reduce(data.overflow.numpy()))
                peaks['nefc'] = max(peaks['nefc'], nefc)
                peaks['nacon'] = max(peaks['nacon'], nacon)
                peaks['solver_niter'] = max(peaks['solver_niter'], niter)
                samples.append({'nefc_max_per_world': nefc, 'nacon_all_worlds': nacon,
                                'solver_niter_max': niter})
            # Timed separately, with no host reads inside the window, so the
            # number is comparable to a training rollout's collection time.
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.steps):
                env.step(action)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        flags = [name for name, bit in
                 ((f.name, int(f.value)) for f in OverflowType) if overflow & bit]
        report = {
            'task': args.task, 'num_envs': env.num_envs, 'measured_control_steps': args.steps,
            'warmup_control_steps': args.warmup, 'decimation': cfg.decimation,
            'physics_dt_s': cfg.sim.mujoco.timestep,
            'allocated': {'naconmax': int(data.naconmax), 'njmax': int(data.njmax),
                          'njmax_nnz': int(data.njmax_nnz), 'nv': int(env.sim.wp_model.nv)},
            'configured': {'nconmax': cfg.sim.nconmax, 'njmax': cfg.sim.njmax,
                           'contact_sensor_maxmatch': cfg.sim.contact_sensor_maxmatch,
                           'iterations': cfg.sim.mujoco.iterations,
                           'ls_iterations': cfg.sim.mujoco.ls_iterations,
                           'ccd_iterations': cfg.sim.mujoco.ccd_iterations},
            'capacity_overrides': overrides or None,
            'timing': {'measured_steps': args.steps, 'seconds_total': elapsed,
                       'seconds_per_control_step': elapsed / args.steps,
                       'env_steps_per_second': env.num_envs * args.steps / elapsed,
                       'gpu': torch.cuda.get_device_name(),
                       'note': ('Zero-action stepping with no policy, PPO or logger. '
                                'Comparable to Perf/collection_time only in order of '
                                'magnitude; compare two runs of this script to each other, '
                                'and only on an otherwise idle GPU.')},
            'measured_peak': peaks,
            'measured_peak_contacts_per_world_mean': peaks['nacon'] / env.num_envs,
            'overflow_bitmask': overflow, 'overflow_flags': flags,
            'samples': samples,
            'interpretation': (
                'nefc is a hard per-world cap: njmax must exceed its peak. nacon is a '
                'total across worlds, so nconmax is a per-world average budget, not a cap. '
                'Any overflow flag invalidates the physics of the affected worlds; a clean '
                'run over this window does not certify unseen states. Zero-action standing '
                'is not the contact-richest state a trained or falling policy reaches, so '
                'keep headroom above these peaks. njmax also sizes the sparse '
                'constraint Jacobian as njmax_nnz = njmax * nv per world, so it is '
                'the single largest allocation lever here. nccdmax defaults to '
                'nconmax and is not exposed by MJLab SimulationCfg, so it cannot be '
                'lowered separately without changing the shared dependency.'),
            'scope': ('GPU physics advanced with zero actions. No training, no checkpoint, '
                      'no capacity was changed by this script.'),
        }
        path = args.output or ROOT / 'outputs' / subdir / 'compute_budget.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
        summary = dict(report)
        summary.pop('samples')
        print(json.dumps(summary, indent=2))
        print(f'[Compute budget] written to {path}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
