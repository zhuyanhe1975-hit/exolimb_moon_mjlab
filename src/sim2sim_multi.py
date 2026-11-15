"""Batched PhysX humans on one Moon; independent local native sensor states."""
from contextlib import ExitStack
from pathlib import Path
import json
import sys
import time
import numpy as np
from sim2sim_contract import ROOT,load_manifest,timing,action_torque,height
from sim2sim_instances import instance_manifest,sample_root


def build_shared_scene(stage,bundle,manifest,skin=True):
    """USD/native CPU authoring only; also usable by the Kit-less audit."""
    import mujoco
    from pxr import UsdGeom,UsdPhysics,Gf
    from sim2sim_usd import build
    from sim2sim_sensors import NativeSensors
    from sim2sim_skin import SuitDisplay
    if manifest.get('instances',{}).get('layout_revision')!='shared_patch_v1':
        raise ValueError('Old multi-map bundle: rerun run_sim2sim.sh to prepare ONE shared moon patch')
    with np.load(Path(bundle)/'instances.npz',allow_pickle=False) as archive:
        bank={key:archive[key] for key in archive.files}
    count=len(bank['roots']);topologies=[];sensors=[];suits=[];manifests=[]
    for i in range(count):
        item=instance_manifest(manifest,i,bank['grids'],bank['roots']);manifests.append(item)
        robot_path=f'/World/Robots/env_{i}';ground_path='/World/Ground'
        topology=build(stage,item,bank['grids'][i],robot_path=robot_path,ground_path=ground_path,scene=i==0,ground=i==0)
        topology['robot_path']=robot_path;topology['ground_path']=ground_path;topologies.append(topology)
        offset=bank['offsets'][i];root=bank['roots'][i]
        # Give every link the SAME rigid transform before PhysX warmup, rather
        # than moving only its root and leaving inconsistent joint anchors.
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,root[3:])
        translation=root[:3]+offset-rotation.reshape(3,3)@np.array(item['bodies'][0]['pos'])
        parent=UsdGeom.Xformable(stage.GetPrimAtPath(robot_path))
        parent.AddTranslateOp().Set(Gf.Vec3d(*map(float,translation)))
        parent.AddOrientOp().Set(Gf.Quatf(float(root[3]),Gf.Vec3f(*map(float,root[4:]))))
        sensor=NativeSensors(bundle,item,skin=skin);sensor.grid=bank['grids'][i]
        sensor.update(root[:3],root[3:],np.zeros(3),np.zeros(3),np.array(item['q_default']),np.zeros(23))
        sensors.append(sensor)
        if skin:suits.append(SuitDisplay(stage,sensor,robot_path=robot_path,suit_path=f'/World/Suits/env_{i}',offset=offset))
    # Exactly one shared ground. Exclude all human/human shoe pairs natively.
    group=UsdPhysics.CollisionGroup.Define(stage,'/World/HumanFeet')
    colliders=[topology['body_paths'][g['body']]+'/'+g['name'] for topology in topologies
               for g in manifest['geoms'] if g['collision']]
    group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(colliders)
    group.CreateFilteredGroupsRel().SetTargets([group.GetPath()])
    return {'bank':bank,'topologies':topologies,'sensors':sensors,'suits':suits,'manifests':manifests}


def run_multi(args,app,startup):
    with ExitStack() as cleanup:
        return _run(args,app,startup,cleanup)


def _run(args,app,startup,cleanup):
    import torch
    import mujoco
    import omni.usd
    from pxr import UsdGeom,UsdLux,Gf
    from isaacsim.core.experimental.prims import Articulation,RigidPrim
    from isaacsim.core.experimental.utils import app as app_utils
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.rendering_manager import RenderingManager
    from sim2sim_skin import update_suits
    # Project namespace import, not another installation's scripts package.
    sys.path.insert(0,str(ROOT))
    from scripts.sim2sim_isaacsim import enable_foot_contact_reporting,validate_foot_contact_views
    manifest=load_manifest(args.bundle)
    if manifest.get('num_envs')!=args.num_envs:raise ValueError('Bundle instance count differs from --num-envs')
    torch.set_num_threads(1);dt,decimation,control_dt=timing(manifest,args.physics_substeps)
    stage=omni.usd.get_context().get_stage();scene=build_shared_scene(stage,args.bundle,manifest,skin=args.skin)
    bank=scene['bank'];sensors=scene['sensors'];topologies=scene['topologies'];suits=scene['suits'];count=len(sensors)
    offsets=bank['offsets'];roots=bank['roots'].copy();rng=np.random.default_rng(manifest['instances']['seed']+1)
    policy=torch.jit.load(str(args.bundle/'policy.pt'),map_location='cpu').eval()
    observation=torch.jit.load(str(args.bundle/'observation.pt'),map_location='cpu').eval()
    from sim2sim_visual import apply_preview_lighting
    apply_preview_lighting(stage)
    camera=None
    if not args.headless:
        from sim2sim_camera import Sim2SimCamera
        camera=Sim2SimCamera(stage,roots[:,:3]+offsets,control_dt,enabled=args.follow);cleanup.callback(camera.close)
    if SimulationManager.get_active_physics_engine()!='physx' and not SimulationManager.switch_physics_engine('physx'):
        raise RuntimeError('PhysX backend unavailable')
    SimulationManager.set_device('cpu');SimulationManager.set_physics_dt(dt)
    SimulationManager.enable_gpu_dynamics(False);SimulationManager.enable_fabric(not args.headless)
    robot=Articulation([item['root_path'] for item in topologies])
    feet=[RigidPrim([item['robot_path']+f'/{side}Foot' for item in topologies],
        contact_filter_paths=[item['ground_path'] for item in topologies],max_contact_count=64*count) for side in ('Left','Right')]
    enable_foot_contact_reporting(feet)
    paths={path:(i,bid) for i,item in enumerate(topologies) for bid,path in item['body_paths'].items()}
    bodies=RigidPrim(list(paths))
    app_utils.play();SimulationManager.initialize_physics();validate_foot_contact_views(feet,dt)
    if not np.isclose(SimulationManager.get_physics_dt(),dt):raise ValueError('PhysX timestep mismatch')
    if robot.num_dofs!=23 or set(robot.dof_names)!=set(manifest['joint_names']):raise ValueError('PhysX joint topology mismatch')
    mapping=np.array([robot.dof_names.index(name) for name in manifest['joint_names']],dtype=np.int32)
    robot.set_dof_gains(stiffnesses=0.,dampings=0.)
    armatures={j['name']:j['armature'] for b in manifest['bodies'] for j in b['joints']}
    expected=np.array([armatures[name] for name in manifest['joint_names']])
    robot.set_dof_armatures(np.tile(expected,(count,1)).astype(np.float32),dof_indices=mapping)
    def array(value):return value.numpy() if hasattr(value,'numpy') else np.asarray(value)
    if not np.allclose(array(robot.get_dof_armatures(dof_indices=mapping)),expected[None],atol=1e-6):
        raise RuntimeError('Batched PhysX armature readback mismatch')
    if any(np.any(abs(array(value))>1e-6) for value in robot.get_dof_gains()):raise RuntimeError('Unexpected implicit drive')
    actions=np.zeros((count,23),dtype=np.float32);loads=np.zeros((count,2));episode=np.zeros(count,dtype=int)
    falls=np.zeros(count,dtype=int);timeouts=np.zeros(count,dtype=int);max_frame_error=0.
    def reset(ids,randomize):
        ids=np.asarray(ids,dtype=np.int32)
        if not len(ids):return
        if randomize:
            for i in ids:
                others=np.delete(robot_positions,i,axis=0)
                roots[i]=sample_root(bank['grids'][i],bank['spawn_xy'][i,:bank['spawn_counts'][i]],bank['foot_points'],bank['padding'][i],rng,np.deg2rad(args.yaw),limit=7.,excluded=others,separation=manifest['instances']['separation_m'])
                robot_positions[i]=roots[i,:3]
        for i in ids:
            sensors[i].target=roots[i,:3].copy();sensors[i].heading=2*np.arctan2(roots[i,6],roots[i,3])
        robot.set_world_poses(positions=(roots[ids,:3]+offsets[ids]).astype(np.float32),orientations=roots[ids,3:].astype(np.float32),indices=ids)
        robot.set_velocities(linear_velocities=np.zeros((len(ids),3),dtype=np.float32),angular_velocities=np.zeros((len(ids),3),dtype=np.float32),indices=ids)
        robot.set_dof_positions(np.tile(manifest['q_default'],(len(ids),1)).astype(np.float32),indices=ids,dof_indices=mapping)
        robot.set_dof_velocities(np.zeros((len(ids),23),dtype=np.float32),indices=ids)
        robot.set_dof_efforts(np.zeros((len(ids),23),dtype=np.float32),indices=ids)
        actions[ids]=0;loads[ids]=0;episode[ids]=0
    def joints():return array(robot.get_dof_positions())[:,mapping],array(robot.get_dof_velocities())[:,mapping]
    def state():
        p,r=map(array,robot.get_world_poses());v,w=map(array,robot.get_velocities());q,qd=joints()
        for i,sensor in enumerate(sensors):sensor.update(p[i].astype(float)-offsets[i],r[i].astype(float),v[i],w[i],q[i],qd[i])
        return p,r,v,w,q,qd
    def touch():
        for side,foot in enumerate(feet):
            force,point,_,_,counts,starts=map(array,foot.get_contact_force_data(dt=dt))
            for i,sensor in enumerate(sensors):
                n,start=int(counts[i,0]),int(starts[i,0])
                if n>=64:raise RuntimeError(f'Contact capacity reached for human {i}, foot {side}')
                loads[i,side]=sensor.touch_load(('left','right')[side],force[start:start+n],point[start:start+n].astype(float)-offsets[i])
    def frames():
        nonlocal max_frame_error
        p,r=map(array,bodies.get_world_poses())
        for row,path in enumerate(bodies.paths):
            i,bid=paths[path];sensor=sensors[i]
            error=float(abs(p[row]-offsets[i]-sensor.data.xpos[bid]).max());max_frame_error=max(max_frame_error,error)
            if error>.005 or abs(np.dot(r[row],sensor.data.xquat[bid]))<.9999:
                raise RuntimeError(f'PhysX/native sensor frame mismatch at human {i}: {error:g}m')
    robot_positions=roots[:,:3].copy()
    reset(np.arange(count),False)
    report={'scope':'N independent PhysX articulations and batched policy inference on ONE 16m moon patch and ONE ground collider; human/human shoe pairs excluded; no MuJoCo dynamics',
        'environments':count,'agent':args.agent,'checkpoint':manifest['checkpoint'],'terrain':manifest['terrain'],
        'instances':manifest['instances'],'physics_dt':dt,'source_physics_dt':manifest['physics_dt'],
        'physics_substeps':args.physics_substeps,'control_dt':control_dt,'gravity_m_s2':manifest['gravity'],
        'pushes':False,'seed':manifest['instances']['seed'],'limitations':manifest['limitations']}
    startup.resume()
    if startup.path:print(f'[Isaac startup log] {startup.path}',flush=True)
    print(f'[Isaac sim2sim] {count} independent humans on ONE 16m moon patch; random separated safe XY/yaw; {1/dt:g}Hz PD / {1/control_dt:g}Hz policy',flush=True)
    print(f'[Checkpoint] {manifest["checkpoint"]}\n[Placement seed] {manifest["instances"]["seed"]}',flush=True)
    print(f'[Display] render every {args.render_every} policy step(s); physics and inference remain at {1/dt:g}Hz and {1/control_dt:g}Hz',flush=True)
    print('[Camera] A=overview/near view; backslash=follow; comma/period=previous/next human; mouse orbit/zoom',flush=True)
    traces=[];obs_log=[];action_log=[];q_log=[];qd_log=[];root_log=[];elapsed=0;last_print=-1
    total=max(1,round(args.seconds/control_dt));length=max(1,round(args.episode_seconds/control_dt));started=time.monotonic()
    rendered=0;last_render_step=0
    with torch.inference_mode():
        for step in range(total):
            if not app.is_running() or not app_utils.is_playing():break
            display_due=not args.headless and ((step+1)%args.render_every==0 or step+1==total)
            p,r,v,w,q,qd=state();raws=[];torso=[]
            for i,sensor in enumerate(sensors):
                raw,tv=sensor.raw(actions[i],loads[i]);raws.append(raw);torso.append(tv)
            obs=observation(torch.from_numpy(np.concatenate(raws)),torch.from_numpy(np.concatenate(torso)))
            if obs.shape!=(count,146) or not torch.isfinite(obs).all():raise FloatingPointError('Invalid batched observation')
            candidate=policy(obs).numpy() if args.agent=='trained' else np.zeros_like(actions)
            if not np.isfinite(candidate).all():raise FloatingPointError('Nonfinite batched policy action')
            peak=np.zeros(count)
            for sub in range(decimation):
                if sub:q,qd=joints()
                effort,clipped,pd=action_torque(candidate,q,qd,manifest);peak=np.maximum(peak,abs(pd).max(-1))
                robot.set_dof_efforts(effort.astype(np.float32),dof_indices=mapping)
                SimulationManager.step(update_fabric=display_due and sub==decimation-1)
            actions[:]=clipped;p,r,v,w,q,qd=state();touch();robot_positions[:]=p-offsets;episode+=1;elapsed+=1
            if step%60==0:frames()
            failed=np.zeros(count,dtype=bool);metrics=[]
            for i,sensor in enumerate(sensors):
                local=p[i].astype(float)-offsets[i];ground=float(height(local[None,:2],sensor.grid,8.)[0])
                tilt=lambda quat:1-2*(quat[1]**2+quat[2]**2)<.5
                failed[i]=(tilt(sensor.value('root_quat')) or tilt(sensor.value('torso_quat')) or
                    local[2]-ground<.7*manifest['nominal_root_height'] or np.linalg.norm(local[:2]-sensor.target[:2])>1 or (abs(local[:2])>6).any())
                a=int(sensor.torso_velocity.adr[0]);speed=np.linalg.norm(sensor.data.sensordata[a:a+3])
                metrics.append([elapsed*control_dt,episode[i]*control_dt,*p[i],np.linalg.norm(sensor.value('torso_omega')),
                    speed,np.linalg.norm(sensor.value('momentum')),np.sqrt(np.mean(qd[i]**2)),*loads[i],int(failed[i]),peak[i]])
            traces.append(metrics);obs_log.append(obs.numpy().copy());action_log.append(actions.copy());q_log.append(q.copy());qd_log.append(qd.copy())
            root_log.append(np.concatenate((p,r,v,w),-1))
            if display_due:
                update_suits(suits)
                if camera is not None:camera.update(p,dt=(step+1-last_render_step)*control_dt)
                RenderingManager.render();rendered+=1;last_render_step=step+1
            timed=(episode>=length)&~failed;falls+=failed;timeouts+=timed
            reset(np.flatnonzero(failed|timed),True)
            second=int(elapsed*control_dt)
            if second%5==0 and second!=last_print:
                print(f'[Inference] t={elapsed*control_dt:.1f}s humans={count} failures={int(falls.sum())}',flush=True);last_print=second
            if args.realtime and not args.headless:
                delay=elapsed*control_dt-(time.monotonic()-started)
                if delay>0:time.sleep(min(delay,control_dt))
    data=np.array(traces,dtype=float).reshape(-1,count,13);quiet=data[:,:,1]>=2
    names=('torso_angular_speed_rad_s','torso_linear_speed_m_s','centroidal_momentum_kg_m2_s','joint_speed_rad_s')
    report.update(simulated_seconds=elapsed*control_dt,seconds_per_env=elapsed*control_dt,requested_seconds=args.seconds,
        episode_seconds=length*control_dt,render_every=args.render_every,rendered_frames=rendered,fall_resets=int(falls.sum()),timeout_resets=int(timeouts.sum()),
        failures_per_env=falls.tolist(),survived_fraction=float((falls==0).mean()),
        survival_passed=elapsed==total and not falls.any(),max_physx_native_frame_error_m=max_frame_error,
        rms_after_2s_each_episode={name:float(np.sqrt(np.mean(data[:,:,5+i][quiet]**2))) if quiet.any() else None for i,name in enumerate(names)})
    path=(args.report or args.bundle/'report.json').resolve();path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),trace=data,
        columns=np.array(['time','episode_time','root_x','root_y','root_z',*names,'left_load_N','right_load_N','failure','max_pd_torque_Nm']),
        observation_before_control=np.array(obs_log),action_applied=np.array(action_log),joint_pos_after_control=np.array(q_log),
        joint_vel_after_control=np.array(qd_log),root_state_after_control=np.array(root_log),joint_names=np.array(manifest['joint_names']))
    path.write_text(json.dumps(report,indent=2)+'\n');print(f'[Report] {path}; failures={int(falls.sum())}',flush=True)
    app_utils.stop()
