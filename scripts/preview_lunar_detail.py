"""Render static native detail previews + metadata; never step physics or learn."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np
import torch
from PIL import Image
from mjlab.scene import Scene
import lunar_finetune_task as task
import local_lunar_terrain as terrain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.patch<terrain.COUNT:parser.error('patch outside 0..63')
    cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=1
    scene=Scene(cfg.scene,'cpu');model=scene.compile()
    ground=terrain.LocalSurface(SimpleNamespace(sim=SimpleNamespace(mj_model=model),num_envs=1,device='cpu'))
    bank=terrain.bank_data();index=args.patch;ground.patch_ids[0]=index
    model.geom_dataid[ground.geom_id]=int(ground.native_ids[index])
    model.geom_pos[ground.geom_id,2]=bank['zmin'][index]
    model.geom_pos[ground.visual_ids.numpy()]=ground.visual_offsets[index].numpy()
    data=mujoco.MjData(model)
    sole=ground.sole_points
    data.qpos[:2]=0
    data.qpos[2]=float((ground.height(sole[None,:,:2])[0]-sole[:,2]).max())+.015+float(ground.clearance_padding[index])
    mujoco.mj_forward(model,data)
    out=Path(__file__).resolve().parents[1]/'outputs/stand_lunar_v15/detail';out.mkdir(parents=True,exist_ok=True)
    options=mujoco.MjvOption()
    with mujoco.Renderer(model,height=768,width=1024) as renderer:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=True
        for name,distance,elevation,lookat in (('standing',4.,-22,[0,0,.8]),('surface',11.,-65,[0,0,0]),('surface_low',5.,-9,[1.5,0,.25])):
            camera=mujoco.MjvCamera();camera.lookat[:]=lookat
            camera.distance=distance;camera.azimuth=135;camera.elevation=elevation
            renderer.update_scene(data,camera=camera,scene_option=options)
            Image.fromarray(renderer.render()).save(out/f'{name}.png')
    detail=bank['detail'];axis=np.linspace(-terrain.HALF,terrain.HALF,detail.shape[1])
    region=(abs(axis)<=4)
    values=detail[:,region,:][:,:,region]
    report={'physics_revision':task.PHYSICS_REVISION,'detail':bank['detail_metadata'],
            'source':bank['source'],'patches':terrain.COUNT,'resolution_m':terrain.STEP,
            'height_delta_m':{'min':float(values.min()),'max':float(values.max()),'rms':float(np.sqrt((values**2).mean()))},
            'preview_patch':index,'scope':'Static native rendering and generated-bank statistics; no simulation or training.'}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
