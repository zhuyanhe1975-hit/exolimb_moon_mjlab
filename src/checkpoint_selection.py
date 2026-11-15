"""Resolve saved runs within one task version, including disturbance variants."""
import re
from pathlib import Path


def latest_saved(directory):
    runs=sorted((p for p in directory.iterdir() if p.is_dir() and 'smoke' not in p.name.lower()),reverse=True) if directory.exists() else []
    for run in runs:
        files=[p for p in run.glob('model_*.pt') if re.fullmatch(r'model_\d+\.pt',p.name)]
        if files:return max(files,key=lambda p:int(p.stem.split('_')[-1]))
    return None


def select_experiment(logs,base,explicit=None,force=None,torque=None):
    if explicit:return explicit,'explicit experiment'
    if force is not None and torque is not None:
        newest=f'{base}_groundsync1_detail3_force{force:g}_torque{torque:g}'
        if latest_saved(logs/newest):return newest,'matching global-field terrain'
        current=f'{base}_groundsync1_detail2_force{force:g}_torque{torque:g}'
        if latest_saved(logs/current):return current,'matching revised detailed terrain'
        detailed=f'{base}_groundsync1_detail1_force{force:g}_torque{torque:g}'
        if latest_saved(logs/detailed):
            return detailed,'matching disturbance experiment, detailed ground physics'
        fixed=f'{base}_groundsync1_force{force:g}_torque{torque:g}'
        if latest_saved(logs/fixed):
            return fixed,'matching disturbance experiment, corrected ground physics'
        matched=f'{base}_force{force:g}_torque{torque:g}'
        if latest_saved(logs/matched):
            return matched,'matching disturbance experiment'
    candidates=[base]
    if logs.exists():
        candidates += [p.name for p in logs.iterdir() if p.is_dir() and
                       (p.name.startswith(base+'_force') or p.name==base+'_no_push' or
                        p.name==base+'_groundsync1' or p.name.startswith(base+'_groundsync1_force') or
                        p.name==base+'_groundsync1_no_push' or p.name==base+'_groundsync1_detail1' or
                        p.name.startswith(base+'_groundsync1_detail1_force') or
                        p.name==base+'_groundsync1_detail1_no_push' or
                        p.name==base+'_groundsync1_detail2' or p.name.startswith(base+'_groundsync1_detail2_force') or
                        p.name==base+'_groundsync1_detail2_no_push' or
                        p.name==base+'_groundsync1_detail3' or p.name.startswith(base+'_groundsync1_detail3_force') or
                        p.name==base+'_groundsync1_detail3_no_push')]
    saved=[(latest_saved(logs/name),name) for name in candidates]
    saved=[(path,name) for path,name in saved if path is not None]
    corrected=[row for row in saved if row[1].startswith(base+'_groundsync1')]
    if corrected:
        saved=corrected
    detailed=[row for row in saved if row[1].startswith((base+'_groundsync1_detail1',base+'_groundsync1_detail2',base+'_groundsync1_detail3'))]
    if detailed:
        saved=detailed
    newest=[row for row in saved if row[1].startswith(base+'_groundsync1_detail3')]
    if newest:
        saved=newest
    revised=[row for row in saved if row[1].startswith(base+'_groundsync1_detail2')]
    if revised:saved=revised
    if saved:
        _,name=max(saved,key=lambda row:(row[0].parent.name,int(row[0].stem.split('_')[-1])))
        return name,'latest saved run within this task version'
    return base,'default experiment (no saved model found)'
