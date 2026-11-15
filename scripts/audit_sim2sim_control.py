"""Static frozen-inertia explicit-PD stability audit. No simulation stepping."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np
from sim2sim_contract import load_manifest,timing


def audit(bundle,output):
    manifest=load_manifest(bundle);model=mujoco.MjModel.from_xml_path(str(bundle/'sensors.xml'))
    data=mujoco.MjData(model);poses=[np.array(manifest['q_default'])]
    if (bundle/'report.npz').is_file():
        with np.load(bundle/'report.npz',allow_pickle=False) as archive:
            qs=archive['joint_pos_after_control']
            poses.extend(qs[i] for i in (0,1,2,5,10,20,40,60,80,100,116) if i<len(qs))
    k=np.diag(manifest['kp']);d=np.diag(np.array(manifest['kd'])+manifest['joint_damping']);eye=np.eye(23)
    profiles=[]
    for subdivisions in (1,2,4,8):
        dt,decimation,period=timing(manifest,subdivisions);radii=[]
        for q in poses:
            data.qpos[:7]=manifest['root_spawn'];data.qpos[7:]=q
            # Native mass matrix includes the armature and floating-root coupling.
            mujoco.mj_fwdPosition(model,data)
            mass=np.empty((model.nv,model.nv));mujoco.mj_fullM(model,data,mass)
            response=np.linalg.solve(mass,np.eye(model.nv))[-23:,-23:]
            velocity=eye-dt*response@d
            transition=np.block([[eye-dt*dt*response@k,dt*velocity],[-dt*response@k,velocity]])
            radii.append(float(np.max(np.abs(np.linalg.eigvals(transition)))))
        profiles.append({'substeps':subdivisions,'physics_hz':1/dt,'policy_hz':1/period,
                         'target_decimation':decimation,'spectral_radius_at_source_pose':radii[0],
                         'max_spectral_radius_over_static_poses':max(radii)})
    report={'scope':'Frozen native inertia, no contacts, no actuator saturation, fixed target. Linear algebra only; NOT a PhysX rollout or standing proof.',
            'source_bundle':str(bundle),'static_poses':len(poses),'profiles':profiles}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/sim2sim_isaacsim/control_audit/report.json'))
    a=p.parse_args();audit(a.bundle.resolve(),a.output.resolve())
