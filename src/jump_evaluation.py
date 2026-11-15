"""Explicit headless evaluation of the single height-calibrated momentum reward experiment."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
import jump_mdp as mdp


def evaluate(args,checkpoint,task,output):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    from push_options import apply
    cfg=apply(task.terrain_stand_cfg(play=False),args.pushes,args.push_force,args.push_torque);cfg.scene.num_envs=args.num_envs
    cfg.episode_length_s=args.seconds+1.
    with output.capture():env=ManagerBasedRlEnv(cfg=cfg,device=args.device)
    try:
        with output.capture():
            wrapper=RslRlVecEnvWrapper(env,clip_actions=task.terrain_stand_rl.clip_actions)
            runner=task.JumpRunner(wrapper,asdict(task.terrain_stand_rl),device=args.device)
            runner.load(str(checkpoint),load_cfg={'actor':True},strict=True,map_location=args.device)
            policy=runner.get_inference_policy(device=args.device)
            obs=wrapper.get_observations()
        sums=torch.zeros(args.num_envs,device=env.device);peaks=sums.clone()
        momentum_sum=sums.clone();momentum_peak=sums.clone();height_peak=sums.clone();airborne_height_peak=sums.clone()
        failures=torch.zeros(args.num_envs,dtype=torch.long,device=env.device)
        count=round(args.seconds/env.step_dt)
        with torch.inference_mode():
            for step in range(count):
                # Current observation-state native sensor values, before any reset.
                up=mdp.upward_speed(env)
                sums+=up;peaks=torch.maximum(peaks,up)
                momentum=mdp.upward_momentum(env)
                momentum_sum+=momentum;momentum_peak=torch.maximum(momentum_peak,momentum)
                obs,_,_,_=wrapper.step(policy(obs))
                failures+=env.reset_terminated.long()
                # Height reward saved these native values BEFORE auto-reset.
                height=env._jump_height_values
                height_peak=torch.maximum(height_peak,height['rise'])
                airborne_height_peak=torch.maximum(airborne_height_peak,height['rise']*height['airborne'])
        report={'task':task.TASK_ID,'checkpoint':str(checkpoint),'num_envs':args.num_envs,
                'seconds_per_env':args.seconds,'physics_revision':task.PHYSICS_REVISION,
                'reward_revision':task.REWARD_REVISION,'jump_height_m':task.JUMP_HEIGHT,
                'momentum_exponent':task.MOMENTUM_EXPONENT,'height_exponent':task.HEIGHT_EXPONENT,
                'max_com_rise_m':height_peak.cpu().tolist(),
                'max_airborne_com_rise_m':airborne_height_peak.cpu().tolist(),
                'target_upward_speed_m_s':mdp.target_upward_speed(env,task.JUMP_HEIGHT),
                'target_upward_momentum_kg_m_s':mdp.target_momentum(env,task.JUMP_HEIGHT).cpu().tolist(),
                'mean_upward_momentum_kg_m_s':(momentum_sum/max(count,1)).cpu().tolist(),
                'max_upward_momentum_kg_m_s':momentum_peak.cpu().tolist(),'pushes':'pelvis_push' in cfg.events,
                'mean_signed_anti_gravity_com_speed_m_s':(sums/max(count,1)).cpu().tolist(),
                'max_upward_com_speed_m_s':peaks.cpu().tolist(),'failure_resets_per_env':failures.cpu().tolist(),
                'scope':'Native COM momentum and measured COM rise from first supported snapshot; not sole clearance or landing success.'}
        path=args.report or Path(__file__).resolve().parents[1]/'outputs'/task.OUTPUT_SUBDIR/f'evaluation_{task.REWARD_REVISION}_k{task.MOMENTUM_EXPONENT:g}_h{task.JUMP_HEIGHT:g}.json'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
        print(f'[Upward momentum evaluation] report={path}',flush=True)
    finally:env.close()
