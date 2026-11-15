"""Bounded CPU physical checks; produces a reproducible report and preview."""
import json
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'assets/wangyu_physics'
OUT=ROOT/'outputs/wangyu_physics';OUT.mkdir(parents=True,exist_ok=True)
m=mujoco.MjModel.from_xml_path(str(ASSET/'scene.xml'));d=mujoco.MjData(m)
mujoco.mj_forward(m,d)
assert m.nu==20 and m.nv==26 and m.nq==27 and m.nmesh==0
assert not any('Neck' in m.joint(i).name or 'Wrist' in m.joint(i).name for i in range(m.njnt))
assert np.all(m.body_inertia[1:]>0)
assert np.all(m.body_inertia[1:].sum(axis=1)[:,None]-2*m.body_inertia[1:] >= -1e-9)
assert abs(m.body_mass.sum()-126.8)<1e-6
report={'nq':m.nq,'nv':m.nv,'actuators':m.nu,'meshes':m.nmesh,'mass_kg':float(m.body_mass.sum()),'gravity':m.opt.gravity.tolist(),'checks':{}}
# Static feasible double-foot wrench, at the global COM projection.
com=d.subtree_com[m.body('Pelvis').id].copy();weight=m.body_mass.sum()*abs(m.opt.gravity[2])
required=d.qfrc_bias.copy();cops=[]
left_y=d.xpos[m.body('LeftFoot').id,1];right_y=d.xpos[m.body('RightFoot').id,1]
left_fraction=(com[1]-right_y)/(left_y-right_y)
assert 0 < left_fraction < 1
for side in ('Left','Right'):
 bid=m.body(side+'Foot').id
 cop=np.array([com[0],d.xpos[bid,1],0.]);cops.append(cop.tolist())
 gid=m.geom(side.lower()+'_sole').id
 local=cop-d.geom_xpos[gid]
 assert np.all(np.abs(local[:2]) <= m.geom_size[gid,:2]+1e-6), 'COM outside support patch'
 jac=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,jr,cop,bid)
 required-=jac.T@np.array([0,0,weight*(left_fraction if side=='Left' else 1-left_fraction)])
assert np.max(np.abs(required[:6]))<1e-6
limits=m.actuator_ctrlrange[:,1];torques=required[m.jnt_dofadr[m.actuator_trnid[:,0]]]
assert np.all(np.abs(torques)<limits)
report['static_support']={'com_world_m':com.tolist(),'weight_n':float(weight),'cop_world_m':cops,'base_wrench_residual':float(np.max(np.abs(required[:6]))),'max_effort_utilization':float(np.max(np.abs(torques)/limits)),'torques_nm':dict(zip([m.actuator(i).name for i in range(m.nu)],torques.tolist()))}
# Exercise each hinge through both limits, elevated away from the ground.
for jid in range(1,m.njnt):
 for angle in (*m.jnt_range[jid],float(m.jnt_range[jid].mean())):
  mujoco.mj_resetData(m,d);d.qpos[2]+=2;d.qpos[m.jnt_qposadr[jid]]=angle;mujoco.mj_forward(m,d)
  assert np.isfinite(d.qacc).all() and np.isfinite(d.xpos).all()
report['checks']['all_joint_limits_finite']=True
# Short unassisted drop confirms native collision response without hidden root support.
mujoco.mj_resetData(m,d);d.qpos[2]+=.15
contacts=set();max_penetration=0.
for step in range(240):
 mujoco.mj_step(m,d)
 assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
 for contact in d.contact:
  if contact.dist<0:
   contacts.update(m.geom(int(i)).name for i in contact.geom)
   max_penetration=max(max_penetration,-float(contact.dist))
assert 'ground' in contacts
assert contacts <= {'ground','left_sole','right_sole'}
report['drop_2s']={'contact_geoms':sorted(contacts),'max_penetration_m':max_penetration,'final_pelvis_height_m':float(d.qpos[2]),'finite':True,'note':'Uncontrolled passive drop, not a standing or learned jumping test.'}
# Check the actuators drive their hinge in the expected direction in free space.
for aid in range(m.nu):
 mujoco.mj_resetData(m,d);d.qpos[2]+=3
 jid=m.actuator_trnid[aid,0];d.qpos[m.jnt_qposadr[jid]]=m.jnt_range[jid].mean()
 mujoco.mj_forward(m,d);base=d.qacc.copy();d.ctrl[aid]=1;mujoco.mj_forward(m,d)
 assert d.qacc[m.jnt_dofadr[jid]]>base[m.jnt_dofadr[jid]]
report['checks']['all_actuator_directions_positive']=True
mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d)
if __name__=='__main__':
 from PIL import Image
 cam=mujoco.MjvCamera();cam.lookat[:]=(0,0,.95);cam.distance=3.5;cam.azimuth=135;cam.elevation=-12
 with mujoco.Renderer(m,height=640,width=800) as renderer:
  renderer.update_scene(d,camera=cam);Image.fromarray(renderer.render()).save(OUT/'native_preview.png')
 (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
