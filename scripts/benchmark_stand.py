"""Bounded matched 8192-world collection A/B, excluding PPO and initialization."""
from pathlib import Path
import os,sys,time,json,statistics
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from stand_task import stand_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends

torch.set_num_threads(1);configure_torch_backends()
cfg=stand_cfg();e=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
try:
    action=torch.zeros((e.num_envs,20),device=e.device)
    values={'eager':[],'cuda_graph':[]}
    with torch.inference_mode():
        for repeat in range(3):
            for mode in (('eager','cuda_graph') if repeat%2==0 else ('cuda_graph','eager')):
                os.environ['WANGYU_TENSOR_GRAPH']='1' if mode=='cuda_graph' else '0'
                e.reset(seed=42)
                for _ in range(32): e.step(action)
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(32): e.step(action)
                torch.cuda.synchronize();duration=time.perf_counter()-start
                if repeat: values[mode].append(duration)
    median={key:statistics.median(value) for key,value in values.items()}
    report={'environments':e.num_envs,'steps_per_environment':32,'physics_dt':cfg.sim.mujoco.timestep,
            'decimation':cfg.decimation,'sim_seconds_per_env_per_rollout':32*e.step_dt,
            'env_control_steps_per_rollout':e.num_envs*32,'physics_step_calls_per_rollout':32*cfg.decimation,
            'physics_cuda_graph':e.sim.use_cuda_graph,'gpu':torch.cuda.get_device_name(),
            'collection_seconds':values,'median_seconds':median,
            'env_steps_per_second':{k:e.num_envs*32/v for k,v in median.items()},
            'graph_time_reduction_fraction':1-median['cuda_graph']/median['eager'],
            'torch_peak_allocated_gb_excluding_warp':torch.cuda.max_memory_allocated()/1e9,
            'scope':'Same env, seeded resets, zero actions, alternating order, 32-step warmup. Initialization, policy inference, PPO update and logger excluded.'}
    path=Path('outputs/stand/performance.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
finally:e.close()
