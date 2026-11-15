"""Independent CPU/static audit. No mj_step, policy inference, environment or GPU.

SciPy hull/segment distances are the geometry oracle. MuJoCo mj_forward is used
only at prescribed static snapshots (zero time integration) to compare native
contact sensors, subtree COM/velocity, and cached versus refreshed contact data.
Does not change production observations/rewards/checkpoints.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import math
import sys
import warnings

import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from support_mdp import signed_margin,compute_geometry,phase_contact_data
import flat_support_task as task
from mjlab.scene import Scene


def oracle(points,query):
    points=np.unique(np.asarray(points,dtype=np.float64),axis=0)
    if len(points)==0:return -.2
    if len(points)==1:return -float(np.linalg.norm(query-points[0]))
    if np.linalg.matrix_rank(points[1:]-points[0],tol=1e-10)<2:
        direction=points[np.argmax(np.linalg.norm(points-points[0],axis=1))]-points[0]
        t=(points-points[0])@direction
        ends=points[[np.argmin(t),np.argmax(t)]]
        a,b=ends;u=b-a
        f=np.clip(np.dot(query-a,u)/np.dot(u,u),0,1)
        return -float(np.linalg.norm(query-a-f*u))
    hull=ConvexHull(points);v=points[hull.vertices];u=np.roll(v,-1,axis=0)-v
    f=np.clip(((query-v)*u).sum(1)/(u*u).sum(1),0,1)
    distance=float(np.linalg.norm(query-v-f[:,None]*u,axis=1).min())
    inside=np.all(hull.equations[:,:2]@query+hull.equations[:,2]<=1e-12)
    return distance if inside else -distance


def random_geometry():
    rng=np.random.default_rng(42);n=3000
    points=rng.normal(size=(n,8,2))*.15;query=rng.normal(size=(n,2))*.2
    mask=rng.random((n,8))>.25;mask[:,:3]=True
    expected=np.array([oracle(p[m],q) for p,m,q in zip(points,mask,query)])
    results={}
    for dtype in (torch.float32,torch.float64):
        actual,_=signed_margin(torch.tensor(points,dtype=dtype),torch.tensor(mask),torch.tensor(query,dtype=dtype))
        error=np.abs(actual.numpy()-expected)
        results[str(dtype)]={'cases':n,'max_error_m':float(error.max()),'errors_over_1e_minus5_m':int((error>1e-5).sum())}
        assert error.max()<1e-6
    return results


def sensor_value(model,data,name):
    sensor=model.sensor('robot/balance_'+name);a=int(sensor.adr[0]);n=int(sensor.dim[0])
    return data.sensordata[a:a+n].copy()


def selected_geometry(contact,com,velocity,gravity):
    mask=((contact.found>0)&(contact.force.norm(dim=-1)>.5)&(contact.dist<=.002)
          &torch.isfinite(contact.pos).all(-1))
    points=torch.where(mask[:,:,None],contact.pos,torch.zeros_like(contact.pos))
    actual=compute_geometry(points,mask,torch.tensor(com)[None],torch.tensor(velocity)[None],gravity)[0].numpy()
    xyz=points[0,mask[0]].numpy();count=len(xyz)
    if count:
        center=xyz.mean(0);height=np.clip(com[2]-center[2],.2,1.5)
        capture=com[:2]+math.sqrt(height/gravity)*velocity[:2]
        static=oracle(xyz[:,:2],com[:2]);dynamic=oracle(xyz[:,:2],capture)
        plane_capture=com[:2]+math.sqrt(np.clip(com[2],.2,1.5)/gravity)*velocity[:2]
        plane_dynamic=oracle(xyz[:,:2],plane_capture)
    else:
        static=dynamic=plane_dynamic=-.2
        capture=None
    assert abs(actual[0]-static)<1e-6 and abs(actual[1]-dynamic)<1e-6
    normal=(contact.found>0)&(contact.force[:,:,0]>.5)&(contact.dist<=.002)
    geometric=(contact.found>0)&(contact.dist<=.002)
    return {'selected_contact_xy_m':xyz[:,:2].tolist(),
            'geometric_contact_xy_m':contact.pos[0,geometric[0],:2].numpy().tolist(),
            'capture_xy_m':None if capture is None else capture.tolist(),
            'selected_contacts':count,'geometric_contacts':int(geometric.sum()),
            'normal_threshold_contacts':int(normal.sum()),'has_area':bool(actual[5]),
            'static_margin_m':float(actual[0]),'capture_proxy_margin_m':float(actual[1]),
            'independent_static_m':float(static),'independent_capture_m':float(dynamic),
            'capture_using_exact_plane_height_m':float(plane_dynamic)}


def native_snapshots():
    cfg=task.terrain_stand_cfg();cfg.scene.num_envs=1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');scene=Scene(cfg.scene,device='cpu');model=scene.compile()
    cfg.sim.mujoco.apply(model)
    sensor=scene['support_contacts']
    buffer=SimpleNamespace(sensordata=torch.zeros((1,model.nsensordata),dtype=torch.float64))
    sensor.initialize(model,None,buffer,'cpu')  # public initialize ignores model argument
    model_body=model.body('robot/Pelvis').id
    output={};native=mujoco.MjData(model)
    poses=[('both_feet',0.,0.,0.),('forward_velocity',0.,.12,0.),
           ('lean',.03,0.,0.),('airborne',0.,0.,.2)]
    previous_contact=None
    for label,pitch,velocity,dz in poses:
        native.qpos[:]=model.qpos0;native.qvel[:]=0
        native.qpos[2]=task.rest_height()-.020+dz
        native.qpos[3:7]=[math.cos(pitch/2),0,math.sin(pitch/2),0]
        native.qvel[0]=velocity;native.ctrl[:]=0
        mujoco.mj_forward(model,native)  # static snapshot only; time remains zero
        assert native.time==0
        com=sensor_value(model,native,'com_pos');vcom=sensor_value(model,native,'com_vel')
        np.testing.assert_allclose(com,native.subtree_com[model_body],atol=1e-12)
        np.testing.assert_allclose(vcom,native.subtree_linvel[model_body],atol=1e-12)
        buffer.sensordata.copy_(torch.tensor(native.sensordata)[None])
        if previous_contact is not None:
            cached=sensor.data
            cached_points=cached.pos.clone()
        contact=phase_contact_data(SimpleNamespace(scene=scene),'observation')  # production fix, dt=0
        result=selected_geometry(contact,com,vcom,abs(model.opt.gravity[2]))
        result['native_contact_count']=native.ncon
        result['com_world_m']=com.tolist();result['com_velocity_world_m_s']=vcom.tolist()
        if previous_contact is not None:
            cached_result=selected_geometry(cached,com,vcom,abs(model.opt.gravity[2]))
            result['capture_margin_using_cached_contacts_m']=cached_result['capture_proxy_margin_m']
            result['production_fixed_capture_margin_m']=result['capture_proxy_margin_m']
            result['phase_mismatch_margin_error_m']=cached_result['capture_proxy_margin_m']-result['capture_proxy_margin_m']
            result['cached_vs_fresh_position_max_delta_m']=float((cached_points-contact.pos).abs().max())
            result['cached_vs_fresh_found_delta']=float((cached.found-contact.found).abs().max())
        previous_contact=contact
        output[label]=result
    # Pure native-buffer phase test: leave ContactSensor's public cache valid,
    # change a declared native position field, observe old .data until update().
    native.qpos[:]=model.qpos0;native.qpos[2]=task.rest_height()-.020;native.qvel[:]=0
    mujoco.mj_forward(model,native)
    buffer.sensordata.copy_(torch.tensor(native.sensordata)[None]);sensor.update(0.)
    primary=sensor.primary_names[0]
    adr=int(model.sensor(f'support_contacts_{primary}_pos').adr[0])
    before=sensor.data.pos.clone();buffer.sensordata[:,adr]+=0.05
    stale=sensor.data.pos.clone();sensor.update(0.);fresh=sensor.data.pos
    output['cache_reproduction']={'cached_data_unchanged':bool(torch.equal(before,stale)),
        'native_buffer_position_change_m':.05,'refreshed_max_position_delta_m':float((fresh-before).abs().max()),
        'production_fix':'phase_contact_data refreshes the stateless public sensor at dt=0 before observation reads.',
        'interpretation':'MJLab Env calls sim.forward/sense after rewards without scene.update when no reset occurs; contact .data can remain pre-forward while COM is post-forward.'}
    assert torch.equal(before,stale) and float((fresh-before).abs().max())>.049
    return output


def plot_snapshots(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,label in zip(axes,('both_feet','forward_velocity','lean')):
        row=result['native_snapshots'][label]
        contacts=np.array(row['selected_contact_xy_m'])
        all_contacts=np.array(row['geometric_contact_xy_m'])
        if len(contacts)>=3 and np.linalg.matrix_rank(contacts[1:]-contacts[0])==2:
            hull=ConvexHull(contacts);polygon=contacts[hull.vertices]
            ax.fill(polygon[:,0],polygon[:,1],alpha=.15,color='tab:blue',label='Loaded contact hull')
            polygon=np.vstack((polygon,polygon[0]));ax.plot(polygon[:,0],polygon[:,1],color='tab:blue')
        ax.scatter(all_contacts[:,0],all_contacts[:,1],facecolors='none',edgecolors='gray',s=65,label='Native contacts')
        ax.scatter(contacts[:,0],contacts[:,1],color='tab:blue',s=18,label='Selected contacts')
        com=np.array(row['com_world_m']);capture=np.array(row['capture_xy_m'])
        ax.scatter(com[0],com[1],marker='x',s=65,color='tab:red',label='CoM projection')
        ax.scatter(capture[0],capture[1],marker='*',s=90,color='purple',label='Capture proxy')
        ax.set_title(label.replace('_',' ')+f"\nmargin={row['capture_proxy_margin_m']*100:.2f} cm")
        ax.set_aspect('equal');ax.set_xlim(-.3,.3);ax.set_ylim(-.3,.3)
        ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc='lower left')
    figure.suptitle('Native static snapshots: independent hull / CoM / capture-point check (no time integration)')
    figure.savefig(out/'static_support.png',dpi=160);plt.close(figure)


def main():
    result={'scope':'CPU numerical oracle + prescribed native static snapshots; no mj_step/GPU/policy/training.',
        'random_geometry':random_geometry(),'native_snapshots':native_snapshots(),
        'limitations':['This is a LIPM capture-point proxy, not a general viability/ZMP test.',
          'Constant COM height and negligible angular momentum rate assumed; external force/torque not included.',
          'Positive-force filtering is an instantaneous loaded subset, not the full available support region.',
          'No-contact margin=-0.2m is a sentinel, not a measured geometric distance.']}
    out=ROOT/'outputs/support_margin_audit';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    plot_snapshots(result,out)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
