"""Static shared-map native rendering; no mj_step, GPU rollout or training."""
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
from shared_moon_viewer import SharedMoonModel


def main():
    count=16;cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=1
    source=Scene(cfg.scene,'cpu').compile();shared=SharedMoonModel(source,count)
    ground=terrain.LocalSurface(SimpleNamespace(sim=SimpleNamespace(mj_model=source),num_envs=count,device='cpu'))
    rng=np.random.default_rng(42);patches=rng.choice(terrain.COUNT,count,replace=False)
    ground.patch_ids[:]=torch.tensor(patches)
    qpos=np.tile(source.qpos0,(count,1));qpos[:,:2]=rng.uniform(-1.25,1.25,(count,2))
    yaw=rng.uniform(-np.pi,np.pi,count);qpos[:,3:7]=0
    qpos[:,3]=np.cos(yaw/2);qpos[:,6]=np.sin(yaw/2)
    foot=ground.sole_points.numpy();c=np.cos(yaw)[:,None];s=np.sin(yaw)[:,None]
    xy=np.stack((qpos[:,None,0]+c*foot[None,:,0]-s*foot[None,:,1],
                 qpos[:,None,1]+s*foot[None,:,0]+c*foot[None,:,1]),-1)
    heights=ground.height(torch.tensor(xy,dtype=torch.float32)).numpy()
    qpos[:,2]=(heights-foot[None,:,2]).max(-1)+.015+ground.clearance_padding[torch.tensor(patches)].numpy()
    shared.update(qpos,np.zeros((count,source.nv)),np.zeros((count,source.nbody,6)),patches)
    model,data=shared.model,shared.data;mujoco.mj_forward(model,data)
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=900
    out=Path(__file__).resolve().parents[1]/'outputs/stand_lunar_v15/shared_view';out.mkdir(parents=True,exist_ok=True)
    positions=data.xpos[shared.pelvis].copy()
    camera=mujoco.MjvCamera();camera.azimuth=135;camera.elevation=-65
    camera.lookat[:]=positions.mean(0);camera.distance=np.linalg.norm(np.ptp(positions,axis=0))*1.2
    with mujoco.Renderer(model,height=900,width=1280) as renderer:
        renderer.update_scene(data,camera=camera)
        Image.fromarray(renderer.render()).save(out/'overview.png')
        camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING;camera.trackbodyid=int(shared.pelvis[0])
        camera.lookat[:]=positions[0];camera.distance=3.5;camera.elevation=-22
        renderer.update_scene(data,camera=camera)
        Image.fromarray(renderer.render()).save(out/'tracked.png')
    (out/'metadata.json').write_text(json.dumps({'humans':count,'terrain_copies':1,
        'native_skins':int(model.nskin),'shared_textures':int(model.ntex),'patches':patches.tolist(),
        'global_pelvis_positions':positions.tolist(),'yaw_rad':yaw.tolist(),
        'scope':'Static shared native display model and original DEM coordinates; no simulation stepping.'},indent=2)+'\n')
    print(f'Static shared view: one map, {count} humans, {model.nskin} independent skins, {model.ntex} shared textures; {out}')


if __name__=='__main__':main()
