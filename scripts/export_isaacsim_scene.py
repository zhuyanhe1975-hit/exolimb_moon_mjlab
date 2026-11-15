"""Export actual native MuJoCo display surfaces to a renderer-only NPZ.

CPU kinematics only: never starts a policy or advances physics. Hfield top
triangles and native mesh transforms preserve the MJLab terrain geometry.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np
from mjlab.scene import Scene
import lunar_finetune_task as task
import local_lunar_terrain as terrain
from terrain_provenance import fingerprint


def export(output):
    cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=1
    model=Scene(cfg.scene,'cpu').compile();data=mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    arrays={};names=[]
    for gid in range(model.ngeom):
        name=model.geom(gid).name
        if name!='site20_full_visual' and not name.startswith('site20_detail_view_'):continue
        asset=int(model.geom_dataid[gid]);kind=model.geom_type[gid]
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            a=int(model.mesh_vertadr[asset]);n=int(model.mesh_vertnum[asset])
            vertices=model.mesh_vert[a:a+n].copy()
            a=int(model.mesh_faceadr[asset]);n=int(model.mesh_facenum[asset])
            faces=model.mesh_face[a:a+n].copy()
        elif kind==mujoco.mjtGeom.mjGEOM_HFIELD:
            rows=int(model.hfield_nrow[asset]);cols=int(model.hfield_ncol[asset])
            a=int(model.hfield_adr[asset]);size=model.hfield_size[asset]
            heights=model.hfield_data[a:a+rows*cols].reshape(rows,cols)*size[2]
            x,y=np.meshgrid(np.linspace(-size[0],size[0],cols),np.linspace(-size[1],size[1],rows))
            vertices=np.stack((x,y,heights),-1).reshape(-1,3)
            base=(np.arange(rows-1)[:,None]*cols+np.arange(cols-1)[None,:]).ravel()
            faces=np.concatenate((np.stack((base,base+1,base+cols+1),-1),
                                  np.stack((base,base+cols+1,base+cols),-1)))
        else:raise ValueError(f'Unsupported terrain geom: {name}')
        vertices=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
        index=len(names);names.append(name)
        arrays[f'points_{index}']=vertices.astype(np.float32);arrays[f'faces_{index}']=faces.astype(np.int32)
    if len(names)!=1:raise ValueError('Incomplete full-map terrain export')
    bank=terrain.bank_data()
    meta={'kind':'static scene preview, no policy replay','physics_revision':task.PHYSICS_REVISION,
          'terrain_fingerprint':fingerprint(),'meshes':names,'meters_per_unit':1,'up_axis':'Z',
          'source_map_offset':np.r_[bank['centers'][0],bank['altitudes'][0]].tolist()}
    model.geom_group[:]=0;model.geom_group[model.geom('terrain').id]=5
    # Native mj_ray skips fully transparent geoms. The play collider is hidden
    # only for display; expose it in this temporary CPU probe model.
    model.geom_rgba[model.geom('terrain').id,3]=1
    meta['ground_heights']={}
    for name,xy in (('actor',(2,0)),('terrain_eye',(0,-5)),('terrain_target',(0,5)),('actor_eye',(4,-3))):
        distance=mujoco.mj_ray(model,data,np.array([*xy,10.]),np.array([0.,0.,-1.]),
            np.array([0,0,0,0,0,1],dtype=np.uint8),True,-1,None)
        if distance<0:raise ValueError('Camera/actor position outside native terrain')
        meta['ground_heights'][name]=float(10-distance)
    arrays['metadata']=np.array(json.dumps(meta))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'[Terrain export] {len(names)} native surfaces, {sum(len(v) for k,v in arrays.items() if k.startswith("points_"))} vertices -> {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args().output)
