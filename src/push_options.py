"""Explicit opt-in for native random pelvis disturbances in all entry points."""
import math
import os


def add_arguments(parser):
    import argparse
    parser.add_argument('--pushes',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--push-force',type=float,default=None)
    parser.add_argument('--push-torque',type=float,default=None)


def resolve(args):
    force=30. if args.push_force is None else args.push_force
    torque=0. if args.push_torque is None else args.push_torque
    if any(not math.isfinite(v) or v<0 for v in (force,torque)):
        raise ValueError('push-force/push-torque must be finite and nonnegative')
    enabled=args.pushes is True or (args.pushes is None and (args.push_force is not None or args.push_torque is not None))
    return enabled,force if enabled else 0.,torque if enabled else 0.


def suffix(enabled,force,torque):
    return f'_force{force:g}_torque{torque:g}' if enabled else '_no_push'


def training(args):
    enabled,force,torque=resolve(args)
    os.environ['WANGYU_TRAIN_PUSH_ENABLED']=str(int(enabled))
    os.environ['WANGYU_TRAIN_PUSH_FORCE']=str(force)
    os.environ['WANGYU_TRAIN_PUSH_TORQUE']=str(torque)
    print(f'[Disturbance] enabled={enabled}; Fx/Fy=±{force:g}N; Fz=0; torque=±{torque:g}Nm (explicit opt-in)',flush=True)
    return enabled,force,torque


def apply(cfg,enabled=False,force=30.,torque=0.):
    if not enabled:
        cfg.events.pop('pelvis_push',None)
    else:
        from stand_push_task import HorizontalBodyImpulse,PUSH_PARAMS
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        cfg.events['pelvis_push']=EventTermCfg(func=HorizontalBodyImpulse,mode='step',params={
            **PUSH_PARAMS,'force_range':(-force,force),'torque_range':(-torque,torque),
            'asset_cfg':SceneEntityCfg('robot',body_names=('Pelvis',))})
    return cfg
