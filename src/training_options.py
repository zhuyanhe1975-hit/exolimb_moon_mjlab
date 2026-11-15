"""Project-local training overrides, parsed before task registration."""
import argparse
import hashlib
import math
import os
from pathlib import Path


def configure(argv):
    from jump_options import configure as configure_jump
    argv=configure_jump(argv)
    if len(argv)<3 or argv[1]!='train' or argv[2] not in ('Wangyu-Moon-Stand-Push-v8','Wangyu-Moon-Stand-Push-v9','Wangyu-Moon-Stand-Support-v10','Wangyu-Moon-Stand-Support-Contact-v11','Wangyu-Moon-Stand-Support-Flat-v12','Wangyu-Moon-Stand-Flat-v13','Wangyu-Moon-Stand-Flat-v14','Wangyu-Moon-Stand-Lunar-v15'):
        return argv
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pushes',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--push-force',type=float)
    parser.add_argument('--push-torque',type=float)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--terrain-transfer',action='store_true')
    args,rest=parser.parse_known_args(argv[3:])
    if args.terrain_transfer:os.environ['WANGYU_TERRAIN_TRANSFER']='1'
    experiment='wangyu_moon_stand_support_v10_contactsync1' if argv[2].endswith('v10') else 'wangyu_moon_stand_push_v9'
    if argv[2].endswith('v8'):experiment='wangyu_moon_stand_push_v8'
    if argv[2].endswith('v11'):experiment='wangyu_moon_stand_support_contact_v11_ccd1e3_contactsync1'
    if argv[2].endswith('v12'):experiment='wangyu_moon_stand_support_flat_v12_contactsync1_quiet1_torso2'
    if argv[2].endswith('v13'):experiment='wangyu_moon_stand_flat_v13'
    if argv[2].endswith('v14'):experiment='wangyu_moon_stand_flat_v14'
    if argv[2].endswith('v15'):
        experiment='wangyu_moon_stand_lunar_v15_groundsync1_detail3'
        os.environ['WANGYU_TRAIN_EXPERIMENT']=experiment
    from push_options import training,suffix
    try:enabled,force,torque=training(args)
    except ValueError as error:parser.error(str(error))
    experiment+=suffix(enabled,force,torque)
    os.environ['WANGYU_TRAIN_EXPERIMENT']=experiment
    print(f'[Training experiment] {experiment}',flush=True)
    if args.resume_from is not None:
        source=args.resume_from.resolve()
        if argv[2].endswith('v15') and not any(name in str(source) for name in ('wangyu_moon_stand_flat_v14','wangyu_moon_stand_lunar_v15')):
            parser.error('v15 terrain transfer requires a 146-input v14/v15 checkpoint')
        if argv[2].endswith('v14') and 'wangyu_moon_stand_flat_v14' not in str(source):
            parser.error('v14 has 146 inputs; older 143/151-input checkpoints cannot resume directly. Train v14 fresh or select the matching older task.')
        if argv[2].endswith('v13') and 'wangyu_moon_stand_flat_v13' not in str(source):
            parser.error('v13 has 143 input dimensions; old 151-dimension checkpoints cannot resume directly. Train v13 fresh or use stand-v12 for old policies.')
        if argv[2].endswith('v12') and 'wangyu_moon_stand_support_flat_v12' not in str(source):
            parser.error('v12 uses native plane physics; start fresh or resume a v12 checkpoint')
        if argv[2].endswith('v11') and 'wangyu_moon_stand_support_contact_v11' not in str(source):
            parser.error('v11 collision model changed; start fresh or resume a v11 checkpoint')
        if argv[2].endswith('v10') and 'wangyu_moon_stand_support_v10' not in str(source):
            parser.error('v10 observation dimension changed to 151; start a new v10 policy, do not resume v8/v9')
        if not source.is_file(): parser.error(f'Checkpoint does not exist: {source}')
        if any(x.startswith(('--agent.resume','--agent.load-run','--agent.load-checkpoint')) for x in rest):
            parser.error('--resume-from cannot be combined with agent resume/load options')
        root=Path(__file__).resolve().parents[1]
        name='0000_smoke_resume_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
        directory=root/'logs/rsl_rl'/experiment/name
        directory.mkdir(parents=True,exist_ok=True)
        link=directory/source.name
        if not link.exists(): link.symlink_to(source)
        rest+=['--agent.resume','True','--agent.load-run',name,'--agent.load-checkpoint',source.name]
        print(f'[Resume source] {source}',flush=True)
    return argv[:3]+rest
