"""Actual-bank CPU native-ray audit of the viewer LOD, without physics steps."""
import json
from pathlib import Path
import sys
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import mujoco
from mjlab.scene import Scene
import lunar_finetune_task as task
from shared_moon_viewer import SharedMoonModel
from sim2sim_contract import height


def main():
    cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');source=Scene(cfg.scene,'cpu').compile()
        view=SharedMoonModel(source,1,full_detail=False)
    model=view.model;data=view.data;bank=view.bank
    model.geom_group[:]=0
    for i in range(model.ngeom):
        if model.geom(i).name.startswith('site20_view_'):model.geom_group[i]=5
    errors=[]
    for patch in range(len(bank['centers'])):
        view.lod.select([patch]);mujoco.mj_kinematics(model,data)
        local=np.array([[.553,-.431],[.137,.219],[7.999,.553],[-7.999,-.431]])
        grid=bank['normalized'][patch]*bank['height_range'][patch]+bank['zmin'][patch]
        expected=height(local,grid,8.)
        for point,wanted in zip(local,expected):
            xy=point+bank['centers'][patch];alt=bank['altitudes'][patch]
            ray=mujoco.mj_ray(model,data,np.r_[xy,alt+10.],np.array([0.,0.,-1.]),
                np.array([0,0,0,0,0,1],dtype=np.uint8),True,-1,None)
            if ray<0:raise AssertionError(f'Missing native display top at patch {patch}: {point}')
            errors.append(abs(10-ray-wanted))
    assert max(errors)<2e-4,max(errors)
    view.lod.select([0])
    report={'scope':'CPU static scene build/native rays only; no GL rendering, dynamics or FPS benchmark',
        'active_patch_ray_checks':len(errors),'max_height_error_m':float(max(errors)),
        'original_terrain_triangles':view.lod.metadata['original_triangles'],
        'one_human_terrain_triangles':view.lod.triangles(),'skins':model.nskin,
        'fine_patches_visible':int(model.geom_rgba[view.lod.fine,3].sum()),
        'coarse_patches_visible':int(model.geom_rgba[view.lod.far,3].sum()),
        'near_resolution_m':.1,'far_interior_resolution_m':1.,'fine_boundary_resolution_m':.1}
    path=Path('outputs/stand_lunar_v15/view_lod_audit/report.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
