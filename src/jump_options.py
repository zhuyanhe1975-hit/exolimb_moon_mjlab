"""Hop CLI and standing prior resolution, before environment construction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from checkpoint_selection import latest_saved

TASK_ID='Wangyu-Moon-Hop-Lunar-v2'
ROOT=Path(__file__).resolve().parents[1]


def select_prior(logs):
    # Only fixed-ground lunar runs; never silently reuse invalid pre-fix v15.
    candidates=[]
    for directory in logs.glob('wangyu_moon_stand_lunar_v15_groundsync1*'):
        checkpoint=latest_saved(directory)
        if checkpoint is None:continue
        record=checkpoint.parent/'continuation.json'
        if record.exists() and json.loads(record.read_text()).get('invalid_terrain_training',False):continue
        candidates.append(checkpoint)
    if candidates:return max(candidates,key=lambda p:(p.parent.name,int(p.stem.split('_')[-1])))
    candidates=[p for d in logs.glob('wangyu_moon_stand_flat_v14*') if (p:=latest_saved(d))]
    if candidates:return max(candidates,key=lambda p:(p.parent.name,int(p.stem.split('_')[-1])))
    raise ValueError('No corrected lunar v15 or flat v14 standing prior found; supply --standing-prior PATH')


def configure(argv):
    if len(argv)<3 or argv[1:3]!=['train',TASK_ID]:return argv
    parser=argparse.ArgumentParser(add_help=False)
    from push_options import add_arguments,training,suffix
    add_arguments(parser)
    parser.add_argument('--jump-height',type=float,default=.25)
    parser.add_argument('--standing-prior',type=Path)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--terrain-transfer',action='store_true')
    args,rest=parser.parse_known_args(argv[3:])
    if any(x in ('--help','-h') for x in rest):
        print('Hop options: --jump-height METRES (default .25); --standing-prior PATH (default: latest corrected v15 / v14); --resume-from PATH (hop only); --terrain-transfer; --pushes/--no-pushes; --push-force N; --push-torque Nm (default: disabled)')
        return argv[:3]+rest
    if args.standing_prior and args.resume_from:parser.error('Choose --standing-prior for initialization or --resume-from for hop continuation')
    if any(x.startswith(('--agent.resume','--agent.load-run','--agent.load-checkpoint')) for x in rest):
        parser.error('Use --resume-from for hop continuation, or --standing-prior for a new experiment')
    from jump_settings import validate_height,experiment_base
    try:height=validate_height(args.jump_height)
    except ValueError as error:parser.error(str(error))
    os.environ['WANGYU_JUMP_HEIGHT']=str(height)
    try:enabled,force,torque=training(args)
    except ValueError as error:parser.error(str(error))
    import local_lunar_terrain as terrain
    experiment=experiment_base(terrain.REVISION,height)+suffix(enabled,force,torque)
    os.environ['WANGYU_JUMP_EXPERIMENT']=experiment
    os.environ.setdefault('WANGYU_DIAGNOSTIC_DIR',str(ROOT/'outputs/hop_lunar_v2/anomalies'))
    if args.terrain_transfer:os.environ['WANGYU_TERRAIN_TRANSFER']='1'
    if args.resume_from:
        import torch
        source=args.resume_from.resolve()
        if not source.is_file():parser.error(f'Checkpoint does not exist: {source}')
        state=torch.load(source,map_location='cpu',weights_only=False)['actor_state_dict']
        if state['mlp.0.weight'].shape[1]!=146:parser.error('v2 requires 146 inputs; old 148-input hop v1 models cannot resume')
        if not (source.parent/'hop_config.json').exists():parser.error('Missing hop_config.json; this is not a saved hop run')
        os.environ.pop('WANGYU_JUMP_PRIOR',None)
        import local_lunar_terrain as terrain
        if json.loads((source.parent/'hop_config.json').read_text()).get('task')!=TASK_ID:
            parser.error('Resume requires this v2 task; use --standing-prior for v14/v15 initialization')
        name='0000_smoke_resume_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
        directory=ROOT/'logs/rsl_rl'/experiment/name;directory.mkdir(parents=True,exist_ok=True)
        link=directory/source.name
        if not link.exists():link.symlink_to(source)
        rest+=['--agent.resume','True','--agent.load-run',name,'--agent.load-checkpoint',source.name]
        print(f'[Hop resume] {source}',flush=True)
    else:
        try:source=(args.standing_prior or select_prior(ROOT/'logs/rsl_rl')).resolve()
        except ValueError as error:parser.error(str(error))
        if not source.is_file():parser.error(f'Prior does not exist: {source}')
        import torch
        state=torch.load(source,map_location='cpu',weights_only=False)['actor_state_dict']
        if state['mlp.0.weight'].shape[1]!=146 or state['mlp.4.weight'].shape[0]!=23:
            parser.error('Standing prior must have 146 inputs and 23 actions (v14/v15)')
        record=source.parent/'continuation.json'
        if record.exists() and json.loads(record.read_text()).get('invalid_terrain_training',False):
            parser.error('Standing prior marked invalid_terrain_training; choose a corrected-ground or flat v14 model')
        os.environ['WANGYU_JUMP_PRIOR']=str(source)
        print(f'[Standing prior selected] {source}; initialization on CURRENT lunar terrain, not exact terrain replay; latest saved is not a stability certification',flush=True)
    return argv[:3]+rest
