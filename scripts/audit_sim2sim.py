"""Static bridge audit: no Kit, timeline, mj_step, or physical rollout.

Run native mode in the MJLab environment, then --usd in the existing Isaac
environment. The latter also compares MuJoCo 3.12 sensors against 3.11 goldens.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import mujoco
import torch
from sim2sim_contract import load_manifest,action_torque,height
from sim2sim_sensors import NativeSensors


def audit(directory,usd=False):
    manifest=load_manifest(directory);sensors=NativeSensors(directory,manifest,skin=True)
    observation=torch.jit.load(str(directory/'observation.pt')).eval()
    policy=torch.jit.load(str(directory/'policy.pt')).eval()
    report={'scope':'CPU static kinematics, USD authoring and tensor inference; no simulation','mujoco':mujoco.__version__}
    if not usd:
        import balance_mdp
        import yaml
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict
        cp=Path(manifest['checkpoint']);cfg=yaml.load((cp.parent/'params/agent.yaml').read_text(),Loader=yaml.FullLoader)['actor']
        actor=MLPModel(TensorDict({'actor':torch.zeros(1,146)},[1]),{'actor':['actor']},'actor',23,
            **{k:cfg[k] for k in ('hidden_dims','activation','obs_normalization','distribution_cfg')})
        actor.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['actor_state_dict']);actor.eval()
        rng=np.random.default_rng(931);rows=[];raws=[];tvs=[];observations=[];outputs=[];poses=[];quats=[]
        max_feature=0.;max_policy=0.
        for i in range(32):
            p=np.array(manifest['root_spawn'][:3])+rng.uniform(-.05,.05,3)
            r=rng.normal(size=4);r/=np.linalg.norm(r)
            velocity=rng.normal(0,.1,6)
            ranges=sensors.model.jnt_range[1:]
            q=rng.uniform(np.maximum(ranges[:,0],-.25),np.minimum(ranges[:,1],.25));qd=rng.normal(0,.3,23)
            a=rng.uniform(-2,2,23);loads=rng.uniform(0,200,2)
            sensors.update(p,r,velocity[:3],velocity[3:],q,qd)
            raw,tv=sensors.raw(a,loads);raws.append(raw[0]);tvs.append(tv[0])
            obs=observation(torch.from_numpy(raw),torch.from_numpy(tv))
            x=balance_mdp.features(torch.from_numpy(raw),float(sensors.model.body_subtreemass[1])*1.635,1.635)[:,:145]
            tq=torch.from_numpy(sensors.value('torso_quat')[None].astype(np.float32))
            expected=torch.cat((x[:,:95],x[:,97:],balance_mdp.inverse_rotate(tq,torch.from_numpy(tv))/.3),-1)
            max_feature=max(max_feature,float((obs-expected).abs().max()))
            with torch.inference_mode():
                output=policy(obs);reference=actor(TensorDict({'actor':obs},[1]))
            max_policy=max(max_policy,float((output-reference).abs().max()))
            outputs.append(output[0].numpy())
            observations.append(obs[0].detach().numpy());rows.append(np.r_[p,r,velocity,q,qd,a,loads])
            poses.append(sensors.data.xpos.copy());quats.append(sensors.data.xquat.copy())
        assert max_feature<2e-5 and max_policy<2e-6,(max_feature,max_policy)
        np.savez_compressed(directory/'static_goldens.npz',state=np.array(rows),raw=np.array(raws),tv=np.array(tvs),
                            obs=np.array(observations),action=np.array(outputs),xpos=np.array(poses),xquat=np.array(quats))
        # This catches the original actor/actuator order mismatch with unequal
        # torso/leg gains and clipping; source native actuator_force is the oracle.
        data=sensors.data;model=sensors.model
        q=np.zeros(23);qd=np.linspace(-.2,.2,23);a=np.linspace(-2.8,2.8,23)
        effort,clipped,pd=action_torque(a,q,qd,manifest)
        data.qpos[7:]=q;data.qvel[6:]=qd
        target=np.clip(clipped*manifest['action_scale'],np.array(manifest['target_limits'])[:,0],np.array(manifest['target_limits'])[:,1])
        for aid in range(model.nu):data.ctrl[aid]=target[int(model.jnt_qposadr[model.actuator_trnid[aid,0]])-7]
        # Static native actuator transmission/velocity refresh; no integration.
        mujoco.mj_fwdPosition(model,data);mujoco.mj_fwdVelocity(model,data)
        mujoco.mj_fwdActuation(model,data);mujoco.mj_passive(model,data)
        native=np.zeros(23)
        for aid in range(model.nu):native[int(model.jnt_qposadr[model.actuator_trnid[aid,0]])-7]=data.actuator_force[aid]
        np.testing.assert_allclose(pd,native,atol=1e-8)
        np.testing.assert_allclose(effort,native+data.qfrc_passive[6:],atol=1e-8)
        report.update(feature_max_abs_error=max_feature,policy_max_abs_error=max_policy,
                      native_pd_and_passive_torque_max_error=float(np.max(np.abs(effort-native-data.qfrc_passive[6:]))),random_poses=32)
    else:
        from pxr import Usd,UsdGeom,UsdPhysics,Gf
        from sim2sim_usd import build
        from sim2sim_skin import SuitDisplay
        stage=Usd.Stage.CreateInMemory();topology=build(stage,manifest,sensors.grid)
        sensors.update(manifest['root_spawn'][:3],manifest['root_spawn'][3:],np.zeros(3),np.zeros(3),np.zeros(23),np.zeros(23))
        suit=SuitDisplay(stage,sensors)
        stage.GetRootLayer().Export(str(directory/'static_scene.usdc'))
        joints=[UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        moving=[j for j in joints if j.GetPrim().IsA(UsdPhysics.RevoluteJoint)]
        assert {j.GetPrim().GetName() for j in moving}==set(manifest['joint_names'])
        colliders=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(colliders)==3,[str(p.GetPath()) for p in colliders]
        assert all(UsdPhysics.DriveAPI(j.GetPrim(),'angular').GetStiffnessAttr().Get()==0 for j in moving)
        archive=np.load(directory/'static_goldens.npz',allow_pickle=False)
        max_raw=max_obs=max_fk=max_policy=0.
        def transform(pos,quat):
            r=np.empty(9);mujoco.mju_quat2Mat(r,np.array([quat.GetReal(),*quat.GetImaginary()]))
            out=np.eye(4);out[:3,:3]=r.reshape(3,3);out[:3,3]=pos;return out
        for i,row in enumerate(archive['state']):
            p,r,v,w,q,qd,a,loads=np.split(row,[3,7,10,13,36,59,82])
            sensors.update(p,r,v,w,q,qd);raw,tv=sensors.raw(a,loads)
            obs=observation(torch.from_numpy(raw),torch.from_numpy(tv)).detach().numpy()[0]
            max_raw=max(max_raw,float(np.max(np.abs(raw[0]-archive['raw'][i]))),float(np.max(np.abs(tv[0]-archive['tv'][i]))))
            max_obs=max(max_obs,float(np.max(np.abs(obs-archive['obs'][i]))))
            with torch.inference_mode():action=policy(torch.from_numpy(obs[None])).numpy()[0]
            max_policy=max(max_policy,float(np.max(np.abs(action-archive['action'][i]))))
            rot=np.empty(9);mujoco.mju_quat2Mat(rot,r)
            root=np.eye(4);root[:3,:3]=rot.reshape(3,3);root[:3,3]=p
            world={topology['root_path']:root};pending=list(joints)
            while pending:
                progress=False
                for j in list(pending):
                    b0,b1=str(j.GetBody0Rel().GetTargets()[0]),str(j.GetBody1Rel().GetTargets()[0])
                    if b0 not in world:continue
                    f0=transform(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get());f1=transform(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get())
                    rotation=np.eye(4)
                    if j.GetPrim().IsA(UsdPhysics.RevoluteJoint):
                        angle=q[manifest['joint_names'].index(j.GetPrim().GetName())];c,s=np.cos(angle),np.sin(angle)
                        rotation[1:3,1:3]=[[c,-s],[s,c]]
                    world[b1]=world[b0]@f0@rotation@np.linalg.inv(f1);pending.remove(j);progress=True
                if not progress:raise AssertionError('USD joint graph is disconnected or cyclic')
            for bid,path in topology['body_paths'].items():
                max_fk=max(max_fk,float(np.max(np.abs(world[path][:3,3]-sensors.data.xpos[bid]))),
                           float(np.max(np.abs(world[path][:3,:3]-sensors.data.xmat[bid].reshape(3,3)))))
            suit.update()
        assert max_raw<5e-4 and max_obs<5e-4 and max_fk<2e-6 and max_policy<2e-5,(max_raw,max_obs,max_fk,max_policy)
        report.update(topology=topology,cross_version_raw_max_error=max_raw,cross_version_observation_max_error=max_obs,
                      usd_vs_native_kinematics_max_error=max_fk,cross_runtime_policy_max_error=max_policy,
                      skins=len(suit.meshes),colliders=len(colliders),moving_joints=len(moving))
    output=directory/('usd_audit.json' if usd else 'native_audit.json')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--usd',action='store_true')
    a=p.parse_args();audit(a.bundle.resolve(),a.usd)
