"""Kit-less USD/native CPU audit of a multi-human bundle. No physics steps."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from sim2sim_contract import load_manifest,height
from sim2sim_multi import build_shared_scene


def audit(bundle):
    manifest=load_manifest(bundle);stage=Usd.Stage.CreateInMemory()
    scene=build_shared_scene(stage,bundle,manifest,skin=True);bank=scene['bank'];n=len(scene['sensors'])
    cache=UsdGeom.XformCache();max_error=0.;gaps=[];raw=[];torso=[]
    for i,(sensor,topology) in enumerate(zip(scene['sensors'],scene['topologies'])):
        offset=bank['offsets'][i];root=bank['roots'][i]
        for bid,path in topology['body_paths'].items():
            transform=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            position=np.array(transform.Transform(Gf.Vec3d(0,0,0)))
            max_error=max(max_error,float(abs(position-offset-sensor.data.xpos[bid]).max()))
        rotation=np.empty(9)
        import mujoco
        mujoco.mju_quat2Mat(rotation,root[3:])
        points=bank['foot_points']@rotation.reshape(3,3).T+root[:3]
        gap=points[:,2]-height(points[:,:2],bank['grids'][i],8.)
        assert gap.min()>=.0145 and gap.max()<=.20,(i,gap.min(),gap.max())
        gaps.extend(gap);x,v=sensor.raw(np.zeros(23),np.zeros(2));raw.append(x);torso.append(v)
    assert max_error<2e-5,max_error
    observation=torch.jit.load(str(bundle/'observation.pt')).eval();policy=torch.jit.load(str(bundle/'policy.pt')).eval()
    with torch.inference_mode():
        obs=observation(torch.from_numpy(np.concatenate(raw)),torch.from_numpy(np.concatenate(torso)));action=policy(obs)
        expected=torch.cat([policy(obs[i:i+1]) for i in range(n)])
    assert obs.shape==(n,146) and action.shape==(n,23)
    np.testing.assert_allclose(action.numpy(),expected.numpy(),atol=2e-6,rtol=2e-6)
    roots=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints=[p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    suits=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and str(p.GetPath()).startswith('/World/Suits/')]
    assert len(roots)==n and len(joints)==23*n and len(suits)==9*n
    colliders=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders)==2*n+1
    terrain=[p for p in colliders if not str(p.GetPath()).startswith('/World/Robots/')]
    assert [str(p.GetPath()) for p in terrain]==['/World/Ground']
    assert len(set(bank['patches'].tolist()))==1
    positions=bank['roots'][:,:2];distance=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
    np.fill_diagonal(distance,np.inf)
    assert distance.min()>=manifest['instances']['separation_m']
    assert (abs(positions)<=5.6).all()
    group=UsdPhysics.CollisionGroup.Get(stage,'/World/HumanFeet')
    assert group.GetFilteredGroupsRel().GetTargets()==[group.GetPath()]
    assert len(group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets())==2*n
    # Independent skin points/parents: do not reuse one actor's vertex buffer.
    first=scene['suits'][0].meshes[0].GetPointsAttr().Get()
    other=scene['suits'][1].meshes[0].GetPointsAttr().Get()
    assert not np.allclose(np.array(first),np.array(other))
    from sim2sim_camera import Sim2SimCamera
    UsdGeom.Camera.Define(stage,'/World/Camera');camera=Sim2SimCamera.__new__(Sim2SimCamera);camera.stage=stage
    target=bank['roots'][0,:3]+bank['offsets'][0];camera.set_view(target+[3,-4,1.4],target)
    prim=stage.GetPrimAtPath('/World/Camera');coi=prim.GetAttribute('omni:kit:centerOfInterest').Get()
    transform=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    np.testing.assert_allclose(np.array(transform.Transform(coi)),target,atol=1e-8)
    report={'scope':'CPU USD/native kinematics and tensor inference only; no Kit/window/PhysX stepping',
        'environments':n,'articulations':len(roots),'moving_hinges':len(joints),'independent_skin_meshes':len(suits),
        'colliders':len(colliders),'ground_colliders':len(terrain),'unique_patches':len(set(bank['patches'].tolist())),
        'minimum_root_xy_separation_m':float(distance.min()),'max_abs_root_xy_m':float(abs(positions).max()),
        'max_native_vs_usd_position_error_m':max_error,'min_spawn_gap_m':float(min(gaps)),'max_spawn_gap_m':float(max(gaps)),
        'max_batched_policy_error':float((action-expected).abs().max()),'seed':manifest['instances']['seed']}
    (bundle/'multi_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    audit(p.parse_args().bundle.resolve())
