"""Play or independently evaluate the latest standing policy."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from startup_output import StartupOutput, run_play_quiet
from checkpoint_selection import select_experiment
startup_output=StartupOutput(verbose='--verbose-init' in sys.argv or os.environ.get('WANGYU_VERBOSE_INIT')=='1')
with startup_output.capture():
    import torch
    import stand_task
    import terrain_stand_task
    import balance_v3_task
    import balance_task
    import lunar_finetune_task as stand_push_task
    import flat_torso_observation_task as flat_v14_task
    import lunar_finetune_task as lunar_v15_task
    import flat_no_margin_task as flat_v13_task
    import flat_support_task as flat_v12_task
    import support_task as support_v10_task
    import tiled_support_task as contact_v11_task
    import stand_pose_free_task as push_v9_task
    import stand_push_task as push_v8_task
    import lqr_task
    import lqr_flat_task
    import lqr_flat_v6_task
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.scripts.play import PlayConfig
    from mjlab.utils.torch import configure_torch_backends
    from stand_task import StandRunner

ROOT=Path(__file__).resolve().parents[1]

def latest_checkpoint(experiment):
    directory=ROOT/'logs/rsl_rl'/experiment
    runs=sorted((p for p in directory.iterdir() if p.is_dir() and 'smoke' not in p.name.lower()),reverse=True) if directory.exists() else []
    for run in runs:
        files=[p for p in run.glob('model_*.pt') if re.fullmatch(r'model_\d+\.pt',p.name)]
        if files: return max(files,key=lambda p:int(p.stem.split('_')[-1]))
    raise FileNotFoundError(f'No saved model in {directory}. Train this experiment first or explicitly use --agent zero. Older experiments are not used as fallback.')


def configure_pushes(cfg, args, task):
    """Playback-only override; keep training registration and checkpoints intact."""
    if task is stand_push_task:
        from push_options import apply
        apply(cfg,args.pushes,args.push_force,args.push_torque)
    return cfg


def evaluate(args, checkpoint, task):
    configure_torch_backends()
    cfg=task.stand_cfg(play=False) if args.flat else task.terrain_stand_cfg(play=False)
    configure_pushes(cfg,args,task)
    cfg.scene.num_envs=args.num_envs
    cfg.episode_length_s=args.episode_seconds if args.episode_seconds is not None else args.seconds+1.
    startup_output.start()
    try:
        e=ManagerBasedRlEnv(cfg=cfg,device=args.device)
    finally:
        startup_output.resume()
    try:
        startup_output.start()
        wrapper=RslRlVecEnvWrapper(e,clip_actions=task.stand_rl.clip_actions if args.flat else task.terrain_stand_rl.clip_actions)
        obs=wrapper.get_observations()
        if args.agent=='trained':
            runner=getattr(task,'LqrRunner',StandRunner)(wrapper,asdict(task.stand_rl if args.flat else task.terrain_stand_rl),device=args.device)
            runner.load(str(checkpoint),load_cfg={'actor':True},strict=True,map_location=args.device)
            policy=runner.get_inference_policy(device=args.device)
        else:
            policy=lambda observation:torch.zeros((args.num_envs,e.action_manager.total_action_dim),device=e.device)
        startup_output.resume()
        motion=None
        if args.motion_trace:
            from motion_diagnostics import MotionDiagnostics
            motion=MotionDiagnostics(e)
        names=('linear_speed','angular_speed','joint_speed','tilt_deg','xy_drift')
        sums=torch.zeros(5,device=e.device);maxima=torch.zeros(5,device=e.device)
        falls=torch.zeros(args.num_envs,dtype=torch.bool,device=e.device)
        reset_count=torch.zeros((),device=e.device);timeout_count=torch.zeros((),device=e.device);count=0
        quiet_sums=torch.zeros(5,device=e.device)
        support_sums=torch.zeros(4,device=e.device);support_min=torch.full((),float('inf'),device=e.device);support_count=0
        with torch.inference_mode():
            for step in range(round(args.seconds/e.step_dt)):
                obs,_,done,_=wrapper.step(policy(obs))
                failure=e.reset_terminated.bool()
                timeout=e.reset_time_outs.bool()&~failure
                falls|=failure;reset_count+=failure.sum();timeout_count+=timeout.sum()
                d=e.scene['robot'].data
                displacement=stand_task.pose_error(e)
                metrics=torch.stack((d.root_link_lin_vel_b.norm(dim=-1),d.root_link_ang_vel_b.norm(dim=-1),
                                     d.joint_vel.square().mean(-1).sqrt(),
                                     torch.acos((-d.projected_gravity_b[:,2]).clamp(-1,1))*180/math.pi,
                                     displacement[:,:2].norm(dim=-1)),dim=-1)
                maxima=torch.maximum(maxima,metrics.max(0).values)
                if task.TASK_ID.endswith(('Support-v10','Support-Contact-v11','Support-Flat-v12','Stand-Flat-v13','Stand-Flat-v14','Stand-Lunar-v15')):
                    if getattr(task,'SUPPORT_PROXY_ENABLED',True):
                        from support_mdp import data as support_data
                        support=support_data(e,'observation')
                    from standing_diagnostics import settled_mask
                    quiet=settled_mask(e)&(d.body_external_wrench.abs().flatten(1).amax(-1)<1e-6)
                    quiet&=e.sim.data.qfrc_applied.abs().amax(-1)<1e-6
                    moving=(metrics[:,0]>.03)|(metrics[:,1]>.1)|(metrics[:,2]>.1)
                    if motion is not None:motion.record(quiet,done)
                    quiet_sums+=torch.stack((quiet.float().sum(),
                        (metrics[:,0].square()*quiet).sum(),(metrics[:,1].square()*quiet).sum(),
                        (metrics[:,2].square()*quiet).sum(),(quiet&moving).float().sum()))
                    if getattr(task,'SUPPORT_PROXY_ENABLED',True):
                        support_sums+=torch.stack((support[:,1].sum(),(support[:,1]<0).float().sum(),
                                                   (support[:,4]==0).float().sum(),support[:,5].sum()))
                        support_min=torch.minimum(support_min,support[:,1].min());support_count+=args.num_envs
                if step*e.step_dt>=1.:
                    sums+=metrics.square().sum(0);count+=args.num_envs
        rms=(sums/max(count,1)).sqrt().cpu().tolist()
        max_values=maxima.cpu().tolist()
        survived=float((~falls).float().mean())
        report={'task':task.TASK_ID,'agent':args.agent,'checkpoint':str(checkpoint) if checkpoint else None,
                'environments':args.num_envs,'seconds_per_env':args.seconds,'episode_length_s':cfg.episode_length_s,'gravity_m_s2':float(e.sim.mj_model.opt.gravity[2]),
                'survived_fraction':survived,'fall_resets':int(reset_count),
                'time_out_resets':int(timeout_count),'total_resets':int(reset_count+timeout_count),
                'rms_after_first_second':dict(zip(names,rms)),'max_over_entire_evaluation':dict(zip(names,max_values)),
                'criteria':{'no_falls':True,'tilt_rms_deg':3.,'root_linear_speed_rms_m_s':.03,'joint_speed_rms_rad_s':.10,'max_xy_drift_m':.05},
                'passed':survived==1 and rms[3]<3 and rms[0]<.03 and rms[2]<.10 and max_values[4]<.05,
                'scope':'Fixed source pose; random Site20 location/yaw per reset, no pushes; finite-duration simulation only.' if not args.flat else 'Fixed initial pose, no pushes; finite-duration simulation only.'}
        if task.TASK_ID.endswith(('Support-v10','Support-Contact-v11','Support-Flat-v12','Stand-Flat-v13','Stand-Flat-v14','Stand-Lunar-v15')):
            values=(support_sums/max(support_count,1)).cpu().tolist()
            quiet_values=quiet_sums.cpu().tolist()
            samples=quiet_values[0]
            report['settled_unforced']={
                'sample_count':int(samples),'minimum_time_since_force_or_reset_s':2.,
                'linear_speed_rms_m_s':math.sqrt(quiet_values[1]/samples) if samples else None,
                'angular_speed_rms_rad_s':math.sqrt(quiet_values[2]/samples) if samples else None,
                'joint_speed_rms_rad_s':math.sqrt(quiet_values[3]/samples) if samples else None,
                'moving_fraction':quiet_values[4]/samples if samples else None,
                'moving_thresholds':{'linear_m_s':.03,'angular_rad_s':.1,'joint_rms_rad_s':.1}}
            report['failure_resets']=report['fall_resets']
            report['criteria']={'no_failure_resets':True}
            report['passed']=survived==1
            report['passed_scope']='Survival only; does not imply motionless standing.'
            if getattr(task,'SUPPORT_PROXY_ENABLED',True):
                report['support_proxy']={
                    'capture_margin_mean_m':values[0], 'outside_fraction':values[1],
                    'no_loaded_contact_fraction':values[2], 'nondegenerate_support_fraction':values[3],
                    'capture_margin_min_m':float(support_min),
                    'scope':'Projected measured-contact hull + LIPM proxy; not general noncoplanar wrench feasibility. Post-reset states included.'}
            else:
                report['support_proxy_enabled']=False
        if args.lqr:
            from lqr_action import term
            controller=term(e)
            report['lqr']={'file':str(controller.artifact_path),'sha256':controller.sha256,
                           'residual_scale_rad':controller.cfg.residual_scale,
                           'fallback_scale_rad':controller.cfg.fallback_scale,
                           'scope':'Flat-ground local gain with terrain/contact gating; RL residual is zero for agent=zero.'}
        if args.lqr_flat:
            report['scope']='Native infinite plane, calibrated equilibrium reset, no pushes; finite-duration simulation only.'
        if task is stand_push_task:
            report['scope']='Random Site20 location/yaw, random horizontal pelvis forces and three-axis torques, no LQR; finite-duration simulation only.'
            report['pushes_enabled']=args.pushes
            report['push_parameters']={**stand_push_task.PUSH_PARAMS,'force_range':(-args.push_force,args.push_force),'torque_range':(-args.push_torque,args.push_torque)} if args.pushes else None
            if not args.pushes:
                report['scope']='Random Site20 location/yaw, no pushes, no LQR; finite-duration simulation only.'
        path=args.report or ROOT/('outputs/stand' if args.flat else ('outputs/stand_terrain' if args.terrain_v2 else ('outputs/stand_lqr_v5' if args.lqr else ('outputs/stand_balance_v3' if args.balance_v3 else 'outputs/stand_balance_v4'))))/('evaluation.json' if checkpoint else 'zero_baseline.json')
        if args.lqr_flat and args.report is None:
            path=ROOT/('outputs/stand_lqr_flat_v6' if args.lqr_flat_v6 else 'outputs/stand_lqr_flat_v7')/('evaluation.json' if checkpoint else 'zero_baseline.json')
        if task is stand_push_task and args.report is None:
            path=ROOT/('outputs/stand_push_v8' if args.push_v8 else ('outputs/stand_push_v9' if args.push_v9 else ('outputs/stand_support_contact_v11' if args.contact_v11 else 'outputs/stand_support_v10')))/(('evaluation.json' if args.pushes else 'evaluation_no_push.json') if checkpoint else ('zero_baseline.json' if args.pushes else 'zero_baseline_no_push.json'))
        if hasattr(task,'TERRAIN_DESCRIPTION'):
            report['scope']=task.TERRAIN_DESCRIPTION+'; no LQR; '+('random pelvis force/torque pulses.' if args.pushes else 'no pushes.')
            if args.report is None:path=ROOT/'outputs'/task.OUTPUT_SUBDIR/path.name
        if getattr(task,'IS_FLAT',False):
            report['scope']='Native infinite plane, fixed source-pose reset, no LQR; '+('random pelvis force/torque pulses.' if args.pushes else 'no pushes.')
            if args.report is None:path=ROOT/'outputs'/task.OUTPUT_SUBDIR/path.name
        if hasattr(task,'PHYSICS_REVISION'):
            report['physics_revision']=task.PHYSICS_REVISION
        if args.selected_experiment!=task.EXPERIMENT and args.report is None:
            path=path.parent/args.selected_experiment/path.name
        if motion is not None:report['motion_diagnostics']=motion.report(path)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        startup_output.resume()
        e.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose-init',action='store_true',help='Show full MJLab/Warp initialization output.')
    parser.add_argument('--full-detail-view',action='store_true',help='Draw all 64 fine lunar patches; default keeps fine detail near humans and uses far display LOD.')
    version=parser.add_mutually_exclusive_group()
    version.add_argument('--flat-v14',action='store_true',help='146-input native-plane task v14.')
    version.add_argument('--lunar-v15',action='store_true',help='146-input local Site20 terrain transfer v15.')
    version.add_argument('--flat-v13',action='store_true',help='Legacy 143-input no-margin plane task.')
    version.add_argument('--flat-v12',action='store_true',help='Legacy 151-input plane policy with support-margin observations.')
    version.add_argument('--support-v10',action='store_true',help='Legacy Site20 support task v10.')
    version.add_argument('--contact-v11',action='store_true',help='Four-tile sole contact experiment v11.')
    version.add_argument('--push-v9',action='store_true',help='Legacy v9 posture-based standing.')
    version.add_argument('--push-v8',action='store_true',help='Legacy v8 with joint pose penalty.')
    version.add_argument('--flat',action='store_true',help='Legacy flat-ground v1.')
    version.add_argument('--terrain-v2',action='store_true',help='Legacy Site20 v2; default is local Site20 terrain v15.')
    version.add_argument('--balance-v3',action='store_true',help='Legacy 20-axis balance v3.')
    version.add_argument('--balance-v4',action='store_true',help='Legacy 23-axis terrain standing without force pulses.')
    version.add_argument('--lqr',action='store_true',help='23-axis LQR + residual RL v5; zero agent means LQR alone.')
    version.add_argument('--lqr-flat',action='store_true',help='Native-plane LQR v7, invariant equilibrium feedforward.')
    version.add_argument('--lqr-flat-v6',action='store_true',help='Legacy native-plane LQR v6.')
    parser.add_argument('--agent',choices=('trained','zero'),default='trained')
    parser.add_argument('--checkpoint-file',type=Path)
    from terrain_provenance import add_playback_options
    add_playback_options(parser)
    parser.add_argument('--experiment-name',help='Select a custom disturbance experiment log directory.')
    parser.add_argument('--num-envs','--num_envs',dest='num_envs',type=int,default=1)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--viewer',choices=('native','viser','auto'),default='native')
    parser.add_argument('--push-force',type=float,default=30.,metavar='N',
                        help='Playback/evaluation Fx/Fy absolute bound in N (default: 30; explicit flag enables pushes).')
    parser.add_argument('--push-torque',type=float,default=0.,metavar='NM',
                        help='World-frame Tx/Ty/Tz absolute bound in Nm (default: 0; explicit flag enables pushes).')
    parser.add_argument('--pushes',action=argparse.BooleanOptionalAction,default=None,
                        help='Enable/disable random pelvis force pulses (standing tasks only; default: disabled).')
    parser.add_argument('--motion-trace',action='store_true',help='Save native motion and actual reward-rate diagnostics during support-task evaluation.')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--episode-seconds',type=float,help='Evaluation episode length; omit to avoid scheduled resets during the evaluation window.')
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.motion_trace and not args.evaluate:parser.error('--motion-trace requires --evaluate')
    if args.episode_seconds is not None and (not args.evaluate or not math.isfinite(args.episode_seconds) or args.episode_seconds<=0):
        parser.error('--episode-seconds requires --evaluate and a finite positive value')
    global stand_push_task
    if args.push_v8: stand_push_task=push_v8_task
    if args.push_v9: stand_push_task=push_v9_task
    if args.contact_v11: stand_push_task=contact_v11_task
    if args.support_v10: stand_push_task=support_v10_task
    if args.flat_v12: stand_push_task=flat_v12_task
    if args.flat_v13: stand_push_task=flat_v13_task
    if args.lunar_v15: stand_push_task=lunar_v15_task
    if args.flat_v14: stand_push_task=flat_v14_task
    args.lqr_flat=args.lqr_flat or args.lqr_flat_v6
    args.lqr=args.lqr or args.lqr_flat
    if not math.isfinite(args.push_force) or args.push_force<0:
        parser.error('--push-force must be finite and nonnegative')
    if not math.isfinite(args.push_torque) or args.push_torque<0:
        parser.error('--push-torque must be finite and nonnegative')
    if args.num_envs<1 or args.seconds<=1: parser.error('num-envs must be positive; seconds must exceed 1')
    torch.set_num_threads(1)
    if startup_output.path is not None:
        print(f'[Startup log] {startup_output.path} (--verbose-init to show)',flush=True)
    task=stand_task if args.flat else (terrain_stand_task if args.terrain_v2 else (balance_v3_task if args.balance_v3 else (balance_task if args.balance_v4 else stand_push_task)))
    if args.push_v8: task=push_v8_task
    if args.lqr: task=lqr_task
    if args.lqr_flat: task=lqr_flat_task
    if args.lqr_flat_v6: task=lqr_flat_v6_task
    if args.pushes is not None and task is not stand_push_task:
        parser.error('--pushes/--no-pushes applies only to stand v8/v9/v10/v11/v12/v13/v14/v15')
    args.pushes = task is stand_push_task and (args.pushes is True or (args.pushes is None and any(
        arg.split('=')[0] in ('--push-force','--push-torque') for arg in sys.argv[1:])))
    if task is stand_push_task:
        explicit_strength=any(arg.split('=')[0] in ('--push-force','--push-torque') for arg in sys.argv[1:])
        experiment,experiment_selection=select_experiment(ROOT/'logs/rsl_rl',task.EXPERIMENT,args.experiment_name,
            args.push_force if explicit_strength else None,args.push_torque if explicit_strength else None)
    else:
        experiment=args.experiment_name or task.EXPERIMENT
        experiment_selection='explicit experiment' if args.experiment_name else 'task default'
    args.selected_experiment=experiment
    if Path(experiment).name!=experiment or experiment in ('.','..'):
        parser.error('--experiment-name must be a directory name, not a path')
    print(f'[Standing task] {task.TASK_ID}',flush=True)
    print(f'[Experiment logs] {ROOT / "logs/rsl_rl" / experiment}',flush=True)
    print(f'[Experiment selection] {experiment_selection}',flush=True)
    print(f'[Runtime] agent={args.agent}, num_envs={args.num_envs}, device={args.device}, viewer={args.viewer}',flush=True)
    print(f'[Physics] gravity=-{stand_task.GRAVITY:g} m/s^2 (1/6G), {23 if task in (balance_task,stand_push_task,lqr_task,lqr_flat_task,lqr_flat_v6_task) else 20} axes, foot-ground collisions only',flush=True)
    if hasattr(task,'CCD_TOLERANCE'):
        print(f'[CCD] native absolute tolerance={task.CCD_TOLERANCE:g}m; diagnostic contact profile',flush=True)
    print('[Terrain] '+task.TERRAIN_DESCRIPTION if hasattr(task,'TERRAIN_DESCRIPTION') else '[Terrain] native infinite plane; fixed source pose, no LQR' if getattr(task,'IS_FLAT',False) else '[Terrain] native infinite plane; equilibrium reset' if args.lqr_flat else ('[Terrain] flat' if args.flat else '[Terrain] Site20 DEM, 2km x 2km; random XY/yaw each reset; spawn slopes <=15 degrees'),flush=True)
    if hasattr(task,'CONTACT_SYNC_REVISION'):
        print(f'[Contact sync] {task.CONTACT_SYNC_REVISION}: post-forward native contact cache refreshed at dt=0',flush=True)
    print(f'[Observation] {72 if args.flat else task.OBS_DIM} dimensions',flush=True)
    if hasattr(task,'OBSERVATION_DESCRIPTION'):
        print(f'[Observation components] {task.OBSERVATION_DESCRIPTION}',flush=True)
    if task in (balance_task,stand_push_task,balance_v3_task,lqr_task):
        if hasattr(task,'TERRAIN_DESCRIPTION'):
            print('[Spawn] local XY +/-1.25m, yaw +/-pi, native-selected patch, sole samples checked; station radius=1m',flush=True)
        elif getattr(task,'IS_FLAT',False):
            print(f'[Spawn] fixed source pose, zero velocity, sole gap={task.CLEARANCE:g}m; station radius={task.STATION_RADIUS:g}m',flush=True)
        else:
            print(f'[Spawn] map margin={task.SPAWN_MARGIN:g}m; station radius={task.STATION_RADIUS:g}m; native collider and all sole samples checked at reset',flush=True)
        print(f'[Reward weights] {task.REWARD_WEIGHTS}',flush=True)
    if task is stand_push_task:
        print(f'[Disturbance] enabled; pelvis COM, world-frame Fx/Fy in [-{args.push_force:g},{args.push_force:g}]N, Fz=0; '
              f'Tx/Ty/Tz each in [-{args.push_torque:g},{args.push_torque:g}]Nm; '
              'duration=0.10..0.20s, cooldown=2..4s independently per environment; no LQR'
              if args.pushes else '[Disturbance] disabled; no random external force pulses',flush=True)
    if args.lqr_flat:
        print('[Spawn] calibrated LQR equilibrium, zero velocity; no free-fall gap or terrain randomization',flush=True)
        print(f'[Reward weights] {task.REWARD_WEIGHTS}',flush=True)
        if not args.lqr_flat_v6:
            print('[Flat control] equilibrium feedforward retained; C1 authority blending; no command low-pass delay',flush=True)
    if not args.flat and not args.evaluate:
        if args.viewer in ('native','auto'):
            print('[Force visualization] native force arrows + purple torque axis/right-hand arcs (Nm); visible only while applied',flush=True)
        print('[View] track=robot/Pelvis, distance=3.5m, visual_extent=5.25m, near_clip=0.0105m, headlight=on',flush=True)
    try:
        checkpoint=(args.checkpoint_file or latest_checkpoint(experiment)).resolve() if args.agent=='trained' else None
        if checkpoint and not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        if task.TASK_ID.endswith('Stand-Flat-v14') and checkpoint and any(name in str(checkpoint) for name in ('wangyu_moon_stand_flat_v13','wangyu_moon_stand_support_flat_v12')):
            parser.error('v14 has 146 inputs. Train v14 fresh or use stand-v13/stand-v12 for the old policy.')
        if task.TASK_ID.endswith('Stand-Flat-v13') and checkpoint and 'wangyu_moon_stand_support_flat_v12' in str(checkpoint):
            parser.error('v13 has 143 inputs; v12 checkpoints have 151. Train v13 fresh or use ./run.sh stand-v12 for the old policy.')
    except FileNotFoundError as error:
        parser.error(str(error))
    if args.lqr:
        from lqr_artifact import DEFAULT_FILE,load_artifact
        controller=checkpoint.parent/'lqr/controller.npz' if checkpoint else Path(os.environ.get('WANGYU_LQR_FILE',str(DEFAULT_FILE)))
        try:
            _,meta,digest=load_artifact(controller)
        except (FileNotFoundError,ValueError) as error:
            parser.error(str(error))
        os.environ['WANGYU_LQR_FILE']=str(controller.resolve())
        print(f'[LQR controller] {controller.resolve()} | sha256={digest}',flush=True)
        print('[Control] LQR + RL residual, 0.05..0.25rad/unit by authority; contact/tilt/joint/slope gating; zero agent = LQR only',flush=True)
    selection='explicit checkpoint' if args.checkpoint_file else 'latest non-smoke run, largest saved model number'
    if task is lunar_v15_task:
        os.environ['WANGYU_MOON_FULL_DETAIL_VIEW']='1' if args.full_detail_view else '0'
    if task is lunar_v15_task and checkpoint:
        from terrain_provenance import verify
        try:verify(checkpoint,allow_transfer=args.terrain_transfer)
        except ValueError as error:parser.error(str(error))
        if args.terrain_transfer:os.environ['WANGYU_TERRAIN_TRANSFER']='1'
    print(f'[Selection] {selection}' if checkpoint else ('[Selection] LQR only, zero RL residual' if args.lqr else '[Selection] Explicit zero-policy PD baseline; no trained model'),flush=True)
    if checkpoint:
        print(f'[Run directory] {checkpoint.parent}',flush=True)
        print(f'[Checkpoint] {checkpoint}',flush=True)
        print(f'[Model file] {checkpoint.name}',flush=True)
        provenance=checkpoint.parent/'continuation.json'
        if task is lunar_v15_task and provenance.exists():
            record=json.loads(provenance.read_text())
            if record.get('invalid_terrain_training',False):
                print('[Checkpoint warning] trained before ground-pose synchronization fix; '
                      'current collision physics is corrected, policy requires retraining from the flat checkpoint.',flush=True)
    if args.evaluate: evaluate(args,checkpoint,task)
    else:
        play_env_cfg=None
        if task is stand_push_task:
            play_env_cfg=configure_pushes(task.terrain_stand_cfg(play=True),args,task)
        run_play_quiet(startup_output,task.TASK_ID,PlayConfig(agent=args.agent,checkpoint_file=str(checkpoint) if checkpoint else None,
                                     num_envs=args.num_envs,device=args.device,viewer=args.viewer),env_cfg=play_env_cfg)

if __name__=='__main__': main()
