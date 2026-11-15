"""Play the independent lunar hop policy; never falls back to standing weights."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from checkpoint_selection import latest_saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    from push_options import add_arguments,resolve,apply
    add_arguments(parser)
    parser.add_argument('--jump-height',type=float,default=None,help='Target airborne COM rise in metres; default .25 or explicit checkpoint metadata.')
    parser.add_argument('--checkpoint-file',type=Path)
    parser.add_argument('--num-envs','--num_envs',type=int,default=1)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--viewer',choices=('native','viser','auto'),default='native')
    parser.add_argument('--verbose-init',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--report',type=Path)
    from terrain_provenance import add_playback_options
    add_playback_options(parser)
    args=parser.parse_args()
    try:args.pushes,args.push_force,args.push_torque=resolve(args)
    except ValueError as error:parser.error(str(error))
    import math
    if args.num_envs<1:parser.error('num-envs must be positive')
    if not math.isfinite(args.seconds) or args.seconds<=0:parser.error('seconds must be finite and positive')
    import json
    from jump_settings import DEFAULT_JUMP_HEIGHT,validate_height
    selected_height=args.jump_height
    if selected_height is None and args.checkpoint_file:
        metadata=args.checkpoint_file.resolve().parent/'hop_config.json'
        if metadata.is_file():selected_height=json.loads(metadata.read_text()).get('jump_height_m')
    try:args.jump_height=validate_height(DEFAULT_JUMP_HEIGHT if selected_height is None else selected_height)
    except ValueError as error:parser.error(str(error))
    os.environ['WANGYU_JUMP_HEIGHT']=str(args.jump_height)
    from startup_output import StartupOutput,run_play_quiet
    output=StartupOutput(verbose=args.verbose_init)
    with output.capture():
        import torch
        import jump_task as task
        from mjlab.scripts.play import PlayConfig
    torch.set_num_threads(1)
    from checkpoint_selection import select_experiment
    logs=Path(__file__).resolve().parents[1]/'logs/rsl_rl'
    experiment,reason=select_experiment(logs,task.EXPERIMENT_BASE,force=args.push_force if args.pushes else None,torque=args.push_torque if args.pushes else None)
    checkpoint=args.checkpoint_file or latest_saved(logs/experiment)
    if checkpoint is None or not checkpoint.is_file():parser.error('No saved hop model; train hop-moon first or supply --checkpoint-file')
    if not (checkpoint.resolve().parent/'hop_config.json').is_file():parser.error('Checkpoint must belong to the independent hop experiment')
    import json
    if json.loads((checkpoint.resolve().parent/'hop_config.json').read_text()).get('task')!=task.TASK_ID:
        parser.error('Select a 146-input v2 model; old 148-input hop models belong to a different experiment')
    os.environ.setdefault('WANGYU_DIAGNOSTIC_DIR',str(Path(__file__).resolve().parents[1]/'outputs/hop_lunar_v2/anomalies'))
    os.environ.pop('WANGYU_JUMP_PRIOR',None)
    os.environ['WANGYU_TERRAIN_TRANSFER']='1' if args.terrain_transfer else '0'
    print(f'[Hop] {task.TASK_ID}; original 146-input standing configuration + anti-gravity COM momentum reward; target rise={task.JUMP_HEIGHT:g}m',flush=True)
    print(f'[Checkpoint] {checkpoint.resolve()} ({reason})',flush=True)
    print(f'[Disturbance] enabled={args.pushes}; force=±{args.push_force:g}N; torque=±{args.push_torque:g}Nm',flush=True)
    if args.evaluate:
        from jump_evaluation import evaluate
        evaluate(args,checkpoint.resolve(),task,output)
        return
    run_play_quiet(output,task.TASK_ID,PlayConfig(checkpoint_file=str(checkpoint.resolve()),
        num_envs=args.num_envs,device=args.device,viewer=args.viewer),
        env_cfg=apply(task.terrain_stand_cfg(play=True),args.pushes,args.push_force,args.push_torque))


if __name__=='__main__':main()
