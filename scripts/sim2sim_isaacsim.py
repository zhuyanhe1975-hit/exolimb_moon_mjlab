"""Run an exported MJLab policy in Isaac Sim PhysX, never in MuJoCo dynamics."""
import json
from pathlib import Path
import sys
import time
import traceback
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sim2sim_contract import arguments,load_manifest,action_torque,height,timing


def enable_foot_contact_reporting(feet):
    """Author native PhysxContactReportAPI BEFORE the first physics setup.

    RigidPrim(contact_filter_paths=...) allocates a filtered view, but the local
    Isaac Sim 6.1 constructor does not enable reports on the sensor bodies.
    """
    import numpy as np
    for foot in feet:
        foot.set_enabled_contact_tracking(True,threshold=0.)
        enabled=foot.get_enabled_contact_tracking()
        enabled=enabled.numpy() if hasattr(enabled,'numpy') else np.asarray(enabled)
        if not enabled.all():
            raise RuntimeError(f'Native contact reporting is disabled on {foot.paths}')


def validate_foot_contact_views(feet,dt):
    """Catch failed native setup callbacks before accepting any inference step."""
    for foot in feet:
        try:
            if not foot.is_physics_tensor_entity_valid():
                raise RuntimeError('rigid body tensor view is invalid')
            if foot.num_contact_filters!=1:
                raise RuntimeError('expected exactly one ground contact filter')
            foot.get_contact_force_data(dt=dt)
        except Exception as error:
            raise RuntimeError(f'Native foot/ground contact view initialization failed for {foot.paths}; '
                               'see the Isaac startup log for the original setup error') from error


def run(args, app, startup):
    if args.num_envs>1:
        from sim2sim_multi import run_multi
        return run_multi(args,app,startup)
    with ExitStack() as cleanup:
        return run_single(args,app,startup,cleanup)


def run_single(args,app,startup,cleanup):
    import numpy as np
    import torch
    import mujoco
    import omni.usd
    from pxr import UsdGeom,UsdLux,Gf
    from isaacsim.core.experimental.prims import Articulation,RigidPrim
    from isaacsim.core.experimental.utils import app as app_utils
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.rendering_manager import RenderingManager
    from sim2sim_usd import build
    from sim2sim_sensors import NativeSensors
    manifest=load_manifest(args.bundle);torch.set_num_threads(1)
    dt,decimation,control_dt=timing(manifest,args.physics_substeps)
    sensors=NativeSensors(args.bundle,manifest,skin=args.skin)
    policy=torch.jit.load(str(args.bundle/'policy.pt'),map_location='cpu').eval()
    observation=torch.jit.load(str(args.bundle/'observation.pt'),map_location='cpu').eval()
    stage=omni.usd.get_context().get_stage()
    topology=build(stage,manifest,sensors.grid)
    from sim2sim_visual import apply_preview_lighting
    apply_preview_lighting(stage)
    origin=np.array(manifest['root_spawn'][:3]);rootq=np.array(manifest['root_spawn'][3:7])
    sensors.update(origin,rootq,np.zeros(3),np.zeros(3),np.zeros(23),np.zeros(23))
    suit=None
    if args.skin:
        from sim2sim_skin import SuitDisplay
        suit=SuitDisplay(stage,sensors)
    camera=None
    if not args.headless:
        from sim2sim_camera import Sim2SimCamera
        camera=Sim2SimCamera(stage,origin[None],control_dt,enabled=args.follow);cleanup.callback(camera.close)
    if SimulationManager.get_active_physics_engine()!='physx':
        if not SimulationManager.switch_physics_engine('physx'):
            raise RuntimeError('PhysX backend unavailable; refusing another physics engine')
    SimulationManager.set_device('cpu')
    SimulationManager.set_physics_dt(dt)
    SimulationManager.enable_gpu_dynamics(False)
    SimulationManager.enable_fabric(not args.headless)
    robot=Articulation(topology['root_path'])
    feet=[RigidPrim(f'/World/Robot/{side}Foot',contact_filter_paths=['/World/Ground'],max_contact_count=64) for side in ('Left','Right')]
    enable_foot_contact_reporting(feet)
    original_bodies=RigidPrim(list(topology['body_paths'].values()))
    app_utils.play()  # Native setup may take one warmup step; reset below.
    SimulationManager.initialize_physics()
    validate_foot_contact_views(feet,dt)
    if not np.isclose(SimulationManager.get_physics_dt(),dt):
        raise ValueError('Target physics timestep differs from the policy contract')
    if robot.num_dofs!=23 or set(robot.dof_names)!=set(manifest['joint_names']):
        raise ValueError(f'PhysX joint mismatch: {robot.dof_names}')
    mapping=np.array([robot.dof_names.index(name) for name in manifest['joint_names']],dtype=np.int32)
    robot.set_dof_gains(stiffnesses=0.,dampings=0.)
    # One source of armature values: native compiled per-joint metadata.
    armatures={j['name']:j['armature'] for b in manifest['bodies'] for j in b['joints']}
    robot.set_dof_armatures(np.array([[armatures[name] for name in manifest['joint_names']]],dtype=np.float32),dof_indices=mapping)
    def array(x):return x.numpy() if hasattr(x,'numpy') else np.asarray(x)
    actual_armatures=array(robot.get_dof_armatures(dof_indices=mapping))[0]
    expected_armatures=np.array([armatures[name] for name in manifest['joint_names']])
    if not np.allclose(actual_armatures,expected_armatures,atol=1e-6,rtol=1e-5):
        raise RuntimeError(f'PhysX armature readback differs from source: {actual_armatures}')
    stiffness,damping=[array(x) for x in robot.get_dof_gains()]
    if np.any(np.abs(stiffness)>1e-6) or np.any(np.abs(damping)>1e-6):
        raise RuntimeError('Implicit PhysX drives must be disabled in explicit effort mode')
    def reset():
        robot.set_world_poses(positions=origin[None].astype(np.float32),orientations=rootq[None].astype(np.float32))
        robot.set_velocities(linear_velocities=np.zeros((1,3),dtype=np.float32),angular_velocities=np.zeros((1,3),dtype=np.float32))
        robot.set_dof_positions(np.array([manifest['q_default']],dtype=np.float32),dof_indices=mapping)
        robot.set_dof_velocities(np.zeros((1,23),dtype=np.float32))
        robot.set_dof_efforts(np.zeros((1,23),dtype=np.float32))
        return np.zeros(23,dtype=np.float32),np.zeros(2)
    def state():
        pos,quat=robot.get_world_poses();lin,ang=robot.get_velocities()
        q,qd=joint_state()
        p,r,v,w=[array(a)[0].astype(float) for a in (pos,quat,lin,ang)]
        sensors.update(p,r,v,w,q,qd)
        return p,r,v,w,q,qd
    def joint_state():
        return array(robot.get_dof_positions())[0,mapping],array(robot.get_dof_velocities())[0,mapping]
    max_frame_error=0.
    def check_frames():
        nonlocal max_frame_error
        # Validate target-engine articulated coordinates against the native
        # sensor model. Do not silently infer observations from a different pose.
        measured_p,measured_q=[array(x) for x in original_bodies.get_world_poses()]
        path_to_native={path:bid for bid,path in topology['body_paths'].items()}
        ids=[path_to_native[path] for path in original_bodies.paths]
        error=float(np.max(np.abs(measured_p-sensors.data.xpos[ids])))
        alignment=np.abs(np.sum(measured_q*sensors.data.xquat[ids],axis=-1))
        max_frame_error=max(max_frame_error,error)
        if error>.005 or np.any(alignment<.9999):
            raise RuntimeError(f'PhysX/native sensor body-frame mismatch: max position error={error:g}m, min quaternion alignment={alignment.min():g}')
    def touch():
        loads=[]
        for side,foot in zip(('left','right'),feet):
            forces,points,_,_,counts,starts=[array(x) for x in foot.get_contact_force_data(dt=dt)]
            n,start=int(counts.sum()),int(starts[0,0])
            if n>=64:raise RuntimeError('PhysX contact buffer may be full; refusing truncated touch observation')
            loads.append(sensors.touch_load(side,forces[start:start+n],points[start:start+n]))
        return np.array(loads)
    action,loads=reset()
    episode_steps=max(1,round(args.episode_seconds/control_dt));total_steps=max(1,round(args.seconds/control_dt))
    report={'scope':'Isaac Sim 6.1 CPU PhysX transfer, original MuJoCo native kinematic sensors only; no MuJoCo dynamics',
        'checkpoint':manifest['checkpoint'],'checkpoint_sha256':manifest['checkpoint_sha256'],'terrain':manifest['terrain'],
        'source_mujoco':manifest['source_mujoco'],'sensor_mujoco':mujoco.__version__,
        'agent':args.agent,'gravity_m_s2':manifest['gravity'],'physics_dt':dt,'control_dt':control_dt,
        'source_physics_dt':manifest['physics_dt'],'physics_substeps':args.physics_substeps,
        'target_decimation':decimation,'pd_integration':'explicit_effort_substepped',
        'armature_readback_policy_order':actual_armatures.tolist(),
        'episode_seconds':episode_steps*control_dt,'pushes':False,'topology':topology,'limitations':manifest['limitations'],
        'physx_dof_names':robot.dof_names,'policy_joint_to_physx_dof':mapping.tolist(),
        'fall_resets':0,'timeout_resets':0,'completed_episodes':0,'survived_episodes':0}
    startup.resume()
    if startup.path:print(f'[Isaac startup log] {startup.path} (--verbose-init to show)',flush=True)
    print(f'[Checkpoint] {manifest["checkpoint"]}\n[Isaac sim2sim] PhysX CPU; 23 hinges; 146 observations; {1/control_dt:g}Hz policy; {1/dt:g}Hz PD; gravity=-1.635; no pushes',flush=True)
    print(f'[Terrain] {manifest["terrain"]}\n[Conversion] {topology["auxiliary_links"]} serial auxiliary links, added mass={topology["extra_mass_kg"]:g}kg',flush=True)
    print('[DOF mapping] '+json.dumps(dict(zip(manifest['joint_names'],mapping.tolist()))),flush=True)
    print(f'[Display] render every {args.render_every} policy step(s); physics and inference remain at {1/dt:g}Hz and {1/control_dt:g}Hz',flush=True)
    trace=[];observations=[];actions=[];positions=[];velocities=[];root_states=[]
    elapsed=0;episode_step=0;last_print=-1;started=time.monotonic();last_render_step=0;rendered=0
    # Keep a renderer-owned app update from adding uncounted physics steps.
    # RenderingManager.render() explicitly updates Kit WITHOUT stepping physics.
    with torch.inference_mode():
        for step in range(total_steps):
            if not app.is_running():break
            if not app_utils.is_playing():
                report['stopped_by_timeline']=True;break
            display_due=not args.headless and ((step+1)%args.render_every==0 or step+1==total_steps)
            p,r,v,w,q,qd=state()
            raw,tv=sensors.raw(action,loads)
            obs=observation(torch.from_numpy(raw),torch.from_numpy(tv))
            if obs.shape!=(1,146) or not torch.isfinite(obs).all():raise FloatingPointError('Invalid policy observation')
            candidate=policy(obs)[0].numpy() if args.agent=='trained' else np.zeros(23,dtype=np.float32)
            if not np.isfinite(candidate).all():raise FloatingPointError('Nonfinite policy output')
            peak_pd=0.
            for substep in range(decimation):
                if substep:q,qd=joint_state()
                effort,clipped,pd=action_torque(candidate,q,qd,manifest)
                peak_pd=max(peak_pd,float(np.max(np.abs(pd))))
                robot.set_dof_efforts(effort[None].astype(np.float32),dof_indices=mapping)
                SimulationManager.step(update_fabric=display_due and substep==decimation-1)
            action=clipped.astype(np.float32)
            p,r,v,w,q,qd=state();loads=touch();elapsed+=1;episode_step+=1
            if step%60==0:check_frames()
            torso_q=sensors.value('torso_quat');root_q=sensors.value('root_quat')
            tilt=lambda quat:float(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1)))
            z_ground=float(height(p[None,:2],sensors.grid,sensors.half)[0])
            # Match the task's 60deg fall guard and 1m station boundary; source
            # nominal height is saved separately from the spawn's terrain z.
            cause=None
            if tilt(root_q)>np.pi/3 or tilt(torso_q)>np.pi/3:cause='tilted'
            elif p[2]-z_ground<.7*manifest['nominal_root_height']:cause='height'
            elif np.linalg.norm(p[:2]-origin[:2])>1.:cause='station_boundary'
            elif sensors.grid is not None and np.any(np.abs(p[:2])>sensors.half-2):cause='terrain_boundary'
            a=int(sensors.torso_velocity.adr[0]);current_tv=sensors.data.sensordata[a:a+3]
            trace.append([elapsed*control_dt,episode_step*control_dt,*p,
                float(np.linalg.norm(sensors.value('torso_omega'))),float(np.linalg.norm(current_tv)),
                float(np.linalg.norm(sensors.value('momentum'))),float(np.sqrt(np.mean(qd**2))),*loads,
                int(cause is not None),peak_pd])
            observations.append(obs.numpy()[0].copy());actions.append(action.copy())
            positions.append(q.copy());velocities.append(qd.copy())
            root_states.append(np.r_[p,r,v,w])
            if display_due:
                if suit:suit.update()
                if camera is not None:camera.update(p[None],dt=(step+1-last_render_step)*control_dt)
                RenderingManager.render();rendered+=1;last_render_step=step+1
            if cause or episode_step>=episode_steps:
                report['completed_episodes']+=1
                if cause:
                    report['fall_resets']+=1;print(f'[Failure] t={elapsed*control_dt:.2f}s reason={cause}',flush=True)
                else:report['timeout_resets']+=1;report['survived_episodes']+=1
                action,loads=reset();episode_step=0
            second=int(elapsed*control_dt)
            if second%5==0 and second!=last_print:
                print(f'[Inference] t={elapsed*control_dt:.1f}s falls={report["fall_resets"]} torso_omega={np.linalg.norm(sensors.value("torso_omega")):.3f}rad/s loads={loads.round(1).tolist()}N',flush=True);last_print=second
            if args.realtime and not args.headless:
                delay=elapsed*control_dt-(time.monotonic()-started)
                if delay>0:time.sleep(min(delay,control_dt))
    report['simulated_seconds']=elapsed*control_dt;report['requested_seconds']=args.seconds
    report['render_every']=args.render_every;report['rendered_frames']=rendered
    report['max_physx_native_frame_error_m']=max_frame_error
    report['finished_requested_duration']=elapsed==total_steps
    report['survival_passed']=elapsed==total_steps and report['fall_resets']==0
    data=np.array(trace,dtype=float).reshape(-1,13)
    # After-reset transient is excluded from quiet RMS, but never from failures.
    quiet=data[:,1]>=2.
    names=('torso_angular_speed_rad_s','torso_linear_speed_m_s','centroidal_momentum_kg_m2_s','joint_speed_rad_s')
    report['rms_after_2s_each_episode']={name:float(np.sqrt(np.mean(data[quiet,5+i]**2))) if quiet.any() else None for i,name in enumerate(names)}
    path=(args.report or args.bundle/'report.json').resolve();path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),trace=data,
        columns=np.array(['time','episode_time','root_x','root_y','root_z',*names,'left_load_N','right_load_N','failure','max_pd_torque_Nm']),
        observation_before_control=np.array(observations),action_applied=np.array(actions),
        joint_pos_after_control=np.array(positions),joint_vel_after_control=np.array(velocities),
        root_state_after_control=np.array(root_states),
        root_state_columns=np.array(['x','y','z','qw','qx','qy','qz','vx','vy','vz','wx','wy','wz']),
        joint_names=np.array(manifest['joint_names']))
    path.write_text(json.dumps(report,indent=2)+'\n');print(f'[Report] {path}\n'+json.dumps({k:report[k] for k in ('simulated_seconds','fall_resets','survival_passed','rms_after_2s_each_episode')},indent=2),flush=True)
    app_utils.stop()


def main():
    args=arguments()
    if args.prepare_only:raise SystemExit('Use ./run_sim2sim.sh --prepare-only to export without starting Kit')
    if not (args.bundle/'manifest.json').is_file():raise FileNotFoundError('Prepare a bundle with ./run_sim2sim.sh first')
    from startup_output import StartupOutput
    startup=StartupOutput(verbose=args.verbose_init);app=None;exit_code=0
    try:
        startup.start()
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':args.headless,'width':1280,'height':720,'limit_cpu_threads':4,'renderer':'RayTracedLighting'})
        run(args,app,startup)
    except Exception as error:
        exit_code=1
        exception_trace=traceback.format_exc()
        startup.resume()
        # Kit's fast close uses os._exit, so Python may never print a pending
        # exception after finally. Print and persist it BEFORE closing Kit.
        print(exception_trace,file=sys.stderr,flush=True)
        if startup.path:print(f'[Isaac startup log] {startup.path}',flush=True)
        manifest=load_manifest(args.bundle)
        destination=(args.report or args.bundle/'report.json').with_suffix('.failure.json')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps({'scope':'Failed startup/inference; no successful transfer result',
            'checkpoint':manifest['checkpoint'],'bundle':str(args.bundle),
            'error_type':type(error).__name__,'error':str(error),
            'traceback':exception_trace},indent=2)+'\n')
        print(f'[Failure report] {destination}',flush=True)
        raise
    finally:
        startup.resume()
        if app is not None:app.close(exit_code=exit_code)


if __name__=='__main__':main()
