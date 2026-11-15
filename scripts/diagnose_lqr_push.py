"""Bounded flat-ground, zero-RL random horizontal force diagnostic.

Explicit execution runs physics, never training. Record native data at 120Hz;
apply world-frame Torso COM forces with MuJoCo/MJLab's native wrench API.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import play_stand as common


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=12.)
    parser.add_argument('--num-envs',type=int,default=16)
    parser.add_argument('--controller',choices=('lqr','fixed_pd'),default='lqr',
                        help='fixed_pd keeps equilibrium PD and disables only outer LQR feedback')
    parser.add_argument('--output',type=Path,default=common.ROOT/'outputs/lqr_push_diagnostic')
    args=parser.parse_args()
    if not 4<=args.num_envs<=64 or not 10<=args.seconds<=30:
        parser.error('Bounded diagnostic: 4..64 environments, 10..30 seconds')
    torch=common.torch
    torch.set_num_threads(1)
    common.configure_torch_backends()
    cfg=common.lqr_flat_task.terrain_stand_cfg(play=False)
    cfg.scene.num_envs=args.num_envs;cfg.episode_length_s=args.seconds+1.
    cfg.seed=42
    with common.startup_output.capture():
        env=common.ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        wrapper=common.RslRlVecEnvWrapper(env,clip_actions=2.)
    robot=env.scene['robot'];controller=env.action_manager.get_term('position')
    if args.controller=='fixed_pd':
        with torch.inference_mode():controller.K.zero_()
        controller._preview_key=None
        env._lqr_observation_key=None
    torso=robot.find_bodies('Torso')[0]
    model=env.sim.mj_model
    actor_ids=[model.actuator('robot/'+name).id for name in controller.target_names]
    sensor_pos=[int(model.sensor('robot/balance_'+side+'_pos').adr[0])+2 for side in ('left','right')]
    amplitudes=np.resize(np.array([0.,30.,60.,120.]),args.num_envs)
    rng=np.random.default_rng(2026)
    directions=rng.uniform(-np.pi,np.pi,(3,args.num_envs))
    forces=torch.zeros((args.num_envs,1,3),device=env.device)
    torques=torch.zeros_like(forces)
    zero=torch.zeros((args.num_envs,23),device=env.device)
    columns={};offset=0
    for name,width in (('root_velocity',3),('root_angular_velocity_body',3),('root_z',1),
                       ('normal_load',2),('authority',1),('target',23),('torque',23),
                       ('joint_pos',23),('joint_vel',23),('sole_z',2),('external_force',3)):
        columns[name]=[offset,offset+width];offset+=width
    records=[];failures=[]
    original_step=env.sim.step

    def record_step():
        original_step()
        velocity=env.sim.data.qvel[:,controller.vadr]
        pose=env.sim.data.qpos[:,controller.qadr]
        sensors=env.sim.data.sensordata
        records.append(torch.cat((velocity[:,:3],velocity[:,3:6],pose[:,2:3],
            sensors[:,controller.load_adr],controller.applied_alpha[:,None],controller.last_command,
            env.sim.data.actuator_force[:,actor_ids],pose[:,7:],velocity[:,6:],
            sensors[:,sensor_pos],forces[:,0,:]),-1).clone())

    env.sim.step=record_step
    try:
        with torch.inference_mode():
            for step in range(round(args.seconds/env.step_dt)):
                time=step*env.step_dt
                forces.zero_()
                for pulse,start in enumerate((2.,5.,8.)):
                    if start<=time<start+.15:
                        values=np.stack((amplitudes*np.cos(directions[pulse]),
                                         amplitudes*np.sin(directions[pulse]),np.zeros(args.num_envs)),-1)
                        forces[:,0,:]=torch.tensor(values,device=env.device,dtype=torch.float32)
                robot.write_external_wrench_to_sim(forces,torques,body_ids=torso)
                _,_,done,_=wrapper.step(zero)
                failures.append(done.clone())
        trace=torch.stack(records).cpu().numpy()
        done=torch.stack(failures).cpu().numpy()
        args.output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output/'trace.npz',trace=trace,done=done,
                            amplitudes=amplitudes,directions=directions)
        metadata={'controller':args.controller,'seconds':args.seconds,'num_envs':args.num_envs,'seed':42,'force_seed':2026,
            'pulse_starts_s':[2.,5.,8.],'pulse_duration_s':.15,'force_amplitudes_N':amplitudes.tolist(),
            'force_body':'Torso COM, world frame, horizontal only','physics_dt':env.physics_dt,
            'control_dt':env.step_dt,'columns':columns,'joint_names':controller.target_names,
            'controller_sha256':controller.sha256,'task':common.lqr_flat_task.TASK_ID,
            'resets_per_env':done.sum(0).tolist(),
            'sampling':'Native force/sensor fields from latest dynamics stage; qpos/qvel after integration. '
                       'Do not infer causality below one physics sample; no extra forward calls inserted.'}
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps({'output':str(args.output),'samples':len(trace),
                          'resets_per_env':done.sum(0).tolist()},indent=2),flush=True)
    finally:
        env.sim.step=original_step
        try:
            with torch.inference_mode():
                forces.zero_();robot.write_external_wrench_to_sim(forces,torques,body_ids=torso)
        finally:
            env.close()


if __name__=='__main__':main()
