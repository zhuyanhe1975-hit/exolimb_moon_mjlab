"""CPU-only policy/asset export. No ManagerBasedRlEnv, mj_step or Isaac runtime."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sim2sim_contract import arguments, ROOT, height,resolve_terrain_mode


def prepare(args):
    import mujoco
    import numpy as np
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from mjlab.entity import Entity
    import yaml
    import balance_mdp
    import torso_velocity
    import flat_torso_observation_task as flat
    import lunar_finetune_task as moon
    from checkpoint_selection import latest_saved, select_experiment
    task = moon if args.task == 'stand-moon' else flat
    selected, reason = select_experiment(ROOT/'logs/rsl_rl', task.EXPERIMENT, explicit=args.experiment_name)
    checkpoint = args.checkpoint_file.resolve() if args.checkpoint_file else latest_saved(ROOT/'logs/rsl_rl'/selected)
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f'No saved checkpoint for {selected}; specify --checkpoint-file')
    if args.checkpoint_file: reason = 'explicit checkpoint'
    print(f'[Selection] {reason}\n[Checkpoint] {checkpoint}', flush=True)
    terrain_mode,terrain_reason=resolve_terrain_mode(args.terrain,args.task,checkpoint,args.terrain_transfer)
    saved_cfg = checkpoint.parent/'params/agent.yaml'
    if not saved_cfg.is_file(): raise FileNotFoundError(f'Checkpoint runner configuration missing: {saved_cfg}')
    cfg = yaml.load(saved_cfg.read_text(), Loader=yaml.FullLoader)
    actor_cfg = cfg['actor']
    if actor_cfg['class_name'] != 'MLPModel' or actor_cfg.get('cnn_cfg') or actor_cfg.get('rnn_type'):
        raise ValueError('This bridge supports the current feed-forward 146x23 standing policy only')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    actor = MLPModel(TensorDict({'actor':torch.zeros(1,146)}, [1]), {'actor':['actor']}, 'actor',23,
        **{k:actor_cfg[k] for k in ('hidden_dims','activation','obs_normalization','distribution_cfg')})
    actor.load_state_dict(state['actor_state_dict'], strict=True); actor.eval()
    torch.set_num_threads(1)
    args.bundle.mkdir(parents=True, exist_ok=True)
    torch.jit.script(actor.as_jit().eval()).save(str(args.bundle/'policy.pt'))
    # Get JOINT/action ordering from the local Entity API, rather than actuator order.
    entity = Entity(task.terrain_stand_cfg().scene.entities['robot'])
    names = list(entity.joint_names)
    target_ids, targets = entity.find_joints_by_actuator_names(('.*',))
    if targets != names or target_ids != list(range(23)):
        raise ValueError('Unsupported action ordering; bridge must be updated explicitly')
    spec = torso_velocity.robot_spec(); spec.option.gravity = [0,0,-1.635]
    m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_kinematics(m,d)
    if m.nq != 30 or m.nv != 29 or len(names) != 23 or not np.allclose(m.body_ipos[1],0):
        raise ValueError('Unexpected physical topology/root COM; refusing implicit frame conversion')
    ids = [m.joint(n).id for n in names]
    np.testing.assert_array_equal(m.jnt_qposadr[ids],np.arange(7,30))
    np.testing.assert_array_equal(m.jnt_dofadr[ids],np.arange(6,29))
    if not np.allclose(m.qpos0[7:],0) or not np.allclose(m.jnt_pos[ids],0):
        raise ValueError('Serial exporter currently requires co-located zero-reference hinges')
    gains = {}; efforts = {}
    for aid in range(m.nu):
        jid = int(m.actuator_trnid[aid,0]); name = m.joint(jid).name
        np.testing.assert_allclose(m.actuator_gear[aid],[1,0,0,0,0,0])
        kp = float(m.actuator_gainprm[aid,0]); kd = float(-m.actuator_biasprm[aid,2])
        if not np.isclose(m.actuator_biasprm[aid,1],-kp): raise ValueError('Non-PD actuator')
        gains[name] = (kp,kd); efforts[name] = m.actuator_forcerange[aid].tolist()
    terrain_meta = {'mode':'flat','source':'native plane','half_extent_m':None}
    grid = None; half = 8.; spawn_xy = np.zeros(2); padding = 0.
    if terrain_mode == 'current' or (terrain_mode == 'trained' and args.task == 'stand-moon'):
        if terrain_mode == 'current':
            import local_lunar_terrain as terrain
            from terrain_provenance import fingerprint
            bank = terrain.bank_data(); half = terrain.HALF
            terrain_meta = {'mode':'lunar','source':'CURRENT terrain transfer','revision':terrain.REVISION,'fingerprint':fingerprint()}
        else:
            directory = checkpoint.parent/'terrain'
            if not (directory/'metadata.json').is_file():
                raise ValueError(f'No verified terrain snapshot in {directory}. Use --terrain current --terrain-transfer for an explicit transfer test, or --terrain flat for a flat-ground comparison.')
            meta = json.loads((directory/'metadata.json').read_text())
            archive = directory/'bank.npz'
            if hashlib.sha256(archive.read_bytes()).hexdigest() != meta['archive_sha256']:
                raise ValueError('Saved terrain archive hash mismatch')
            if 'groundsync1' not in meta['physics_revision']:
                raise ValueError('Checkpoint predates the GPU ground-position correction; invalid terrain training is not a valid source')
            with np.load(archive,allow_pickle=False) as a: bank = {k:a[k] for k in a.files}
            half = 8.
            terrain_meta = {'mode':'lunar','source':str(directory),'revision':meta['physics_revision'],'fingerprint':meta['fingerprint']}
        if args.patch >= len(bank['normalized']): raise ValueError('Patch index outside saved bank')
        index = args.patch
        grid = bank['normalized'][index].astype(np.float64)*bank['height_range'][index]+bank['zmin'][index]
        # Same 16m patch and triangle heights used by training; do not silently
        # substitute the latest procedural bank for an older trained checkpoint.
        terrain_meta.update(patch=index, half_extent_m=half, resolution_m=2*half/(len(grid)-1))
        if 'spawn_counts' in bank:
            if bank['spawn_counts'][index] == 0: raise ValueError('Selected patch has no certified spawn; select another --patch')
            spawn_xy = np.array(bank['spawn_xy'][index,0],dtype=float)
        dz = np.diff(grid,axis=0)/(2*half/(len(grid)-1))
        dx = np.diff(grid,axis=1)/(2*half/(len(grid)-1))
        # Conservative global slope bound, valid even for historical banks.
        slope = float(np.hypot(np.abs(dx).max(),np.abs(dz).max()))
        terrain_meta['max_slope_bound'] = slope
        from local_lunar_terrain import sole_sampling
        # sole_sampling expects attached robot/ names; attach a native prefix.
        parent = mujoco.MjSpec(); frame=parent.worldbody.add_frame()
        probe = mujoco.MjSpec.from_string(spec.to_xml())
        parent.option.gravity=probe.option.gravity;parent.option.timestep=probe.option.timestep
        parent.option.integrator=probe.option.integrator
        frame.attach_body(probe.body('Pelvis'),prefix='robot/')
        points, cover = sole_sampling(parent.compile())
        padding = slope*cover+2e-6
    else:
        # Dense sole samples in native rest coordinates, including all corners.
        points=[]
        for side in ('left','right'):
            gid=m.geom(side+'_sole').id; size=m.geom_size[gid]
            x,y=np.meshgrid(np.linspace(-size[0],size[0],17),np.linspace(-size[1],size[1],11))
            local=np.stack((x,y,np.full_like(x,-size[2])),-1).reshape(-1,3)
            points.extend(local@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]-d.qpos[:3])
        points=np.array(points)
    yaw=np.deg2rad(args.yaw); c,s=np.cos(yaw),np.sin(yaw)
    rotated=points[:,:2]@np.array([[c,s],[-s,c]])+spawn_xy
    root_z=float(np.max(height(rotated,grid,half)-points[:,2])+.015+padding)
    gaps=root_z+points[:,2]-height(rotated,grid,half)
    if gaps.min()<.0145 or gaps.max()>.20: raise ValueError(f'Invalid sole spawn gaps {gaps.min()}..{gaps.max()}m')
    root=[*spawn_xy,root_z,np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
    terrain_meta['selection_reason']=terrain_reason
    # Export native sensor XML without any dynamics step. Original body IDs,
    # sites, inertias and joint coordinate order remain unchanged.
    (args.bundle/'sensors.xml').write_text(spec.to_xml())
    bodies=[]
    for bid in range(1,m.nbody):
        joints=[j for j in ids if m.jnt_bodyid[j]==bid]
        bodies.append({'id':bid,'name':m.body(bid).name,'parent':int(m.body_parentid[bid]),
            'pos':d.xpos[bid].tolist(),'quat':d.xquat[bid].tolist(),
            'mass':float(m.body_mass[bid]),'com':m.body_ipos[bid].tolist(),
            'inertia':m.body_inertia[bid].tolist(),'inertia_quat':m.body_iquat[bid].tolist(),
            'joints':[{'name':m.joint(j).name,'axis':m.jnt_axis[j].tolist(),'range':m.jnt_range[j].tolist(),
                       'armature':float(m.dof_armature[m.jnt_dofadr[j]])} for j in joints]})
    geoms=[]
    for gid in range(m.ngeom):
        geoms.append({'name':m.geom(gid).name,'body':int(m.geom_bodyid[gid]),'type':int(m.geom_type[gid]),
            'size':m.geom_size[gid].tolist(),'pos':m.geom_pos[gid].tolist(),'quat':m.geom_quat[gid].tolist(),
            'rgba':m.geom_rgba[gid].tolist(),'collision':bool(m.geom_contype[gid] or m.geom_conaffinity[gid])})
    env_cfg=task.terrain_stand_cfg(); acfg=env_cfg.actions['position']
    manifest={'version':1,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_mujoco':mujoco.__version__,'task':task.TASK_ID,'obs_dim':146,'action_dim':23,
        'joint_names':names,'q_default':m.qpos0[7:].tolist(),'root_spawn':root,'terrain':terrain_meta,
        'spawn_gap_min_m':float(gaps.min()),'spawn_gap_max_m':float(gaps.max()),
        'physics_dt':float(env_cfg.sim.mujoco.timestep),'decimation':int(env_cfg.decimation),
        'nominal_root_height':float(env_cfg.scene.entities['robot'].init_state.pos[2]),
        'gravity':-1.635,'action_scale':float(acfg.scale),'clip_actions':float(cfg['clip_actions']),
        'target_limits':[list(acfg.clip[n]) for n in names],
        'kp':[gains[n][0] for n in names],'kd':[gains[n][1] for n in names],
        'effort_limits':[efforts[n] for n in names],'joint_damping':m.dof_damping[6:].tolist(),
        'bodies':bodies,'geoms':geoms,'sensor_layout':balance_mdp.SENSORS,
        'serial_auxiliary_mass_kg':1e-4,'serial_auxiliary_inertia_kg_m2':1e-6,
        'limitations':['PhysX contact/limit/friction solver differs from MuJoCo; this is a transfer test.',
          'Multi-hinge bodies use serial auxiliary links with small positive mass/inertia.',
          'PD and passive damping are explicit at physics rate; source implicit integration differs.',
          'Only normal foot loads inside native TOUCH site volumes are transferred; no synthetic MuJoCo contacts.',
          'Single actor/local terrain patch, no random pushes; no claim of photorealistic or dynamic validation.']}
    np.savez_compressed(args.bundle/'terrain.npz', grid=np.zeros((0,0)) if grid is None else grid)
    # Reuse the production feature function itself, including ordering/scales.
    class FeatureModule(torch.nn.Module):
        def forward(self, raw, torso_velocity):
            x=balance_mdp.features(raw, float(m.body_subtreemass[1])*1.635,1.635)[:,:145]
            tq=raw[:,92+13:92+17]
            linear=balance_mdp.inverse_rotate(tq,torso_velocity)/.3
            return torch.cat((x[:,:95],x[:,97:],linear),-1)
    width=balance_mdp.STATE_DIM+balance_mdp.SENSOR_DIM+30
    example=torch.zeros(1,width); example[:,95]=1.; example[:,105]=1.
    torch.jit.trace(FeatureModule().eval(),(example,torch.zeros(1,3)),check_trace=True).save(str(args.bundle/'observation.pt'))
    # Optional native deformed suit; no alternate skeleton or retargeting.
    skin=torso_velocity.skin_spec()
    skin.option.gravity=[0,0,-1.635]
    # Standalone XML does not use the robot/ attachment material namespace.
    for item in skin.skins: item.material=item.material.removeprefix('robot/')
    for tex in skin.textures:
        if tex.file: tex.file=str(ROOT/'assets'/tex.file)
    (args.bundle/'skin.xml').write_text(skin.to_xml())
    manifest['num_envs']=args.num_envs
    if args.num_envs>1:
        if terrain_mode!='current':raise ValueError('Shared multi-human scene requires --terrain moon')
        from sim2sim_instances import prepare_instances
        prepare_instances(args,manifest,bank,points,cover)
    (args.bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'[Bundle] {args.bundle}\n[Contract] 146 observations, 23 JOINT-order actions; control={1/(manifest["physics_dt"]*manifest["decimation"]):g}Hz',flush=True)
    print(f'[Terrain] {terrain_meta}\n[Spawn] sole gap={gaps.min():.5f}..{gaps.max():.5f}m; gravity=-1.635',flush=True)
    print('[Scope] CPU export completed; no physics or policy rollout executed',flush=True)
    return manifest


if __name__=='__main__':
    from startup_output import StartupOutput
    args=arguments();startup=StartupOutput(verbose=args.verbose_init)
    try:
        with startup.capture():manifest=prepare(args)
    finally:
        if startup.path:print(f'[Export log] {startup.path} (--verbose-init to show)',flush=True)
    if startup.enabled:
        print(f'[Checkpoint] {manifest["checkpoint"]}\n[Bundle] {args.bundle}',flush=True)
        print(f'[Contract] 146 observations, 23 axes; gravity=-1.635; {1/(manifest["physics_dt"]*manifest["decimation"]):g}Hz policy',flush=True)
        print(f'[Terrain] {manifest["terrain"]}\n[CPU export] no physics stepping or policy rollout',flush=True)
        if args.num_envs>1:print(f'[Instances] {args.num_envs} humans on ONE 16m moon patch; random safe XY/yaw; seed={manifest["instances"]["seed"]}',flush=True)
