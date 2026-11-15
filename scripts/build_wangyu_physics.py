"""Build a mesh-free Wangyu MJCF with Apollo-inspired fixed-head articulation.
No external asset dependency at runtime. SI units; +X forward, +Y left, +Z up.
"""
import json
import math
import hashlib
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/wangyu_physics'

def fmt(v): return ' '.join(f'{x:.9g}' for x in np.asarray(v).ravel())
def build():
    source=json.loads((OUT/'rig_measurements.json').read_text())
    a={k:np.array(v) for k,v in source['anchors'].items()}
    # Preserve the source posed rig, including its arm spread and knee bend.
    # Center the floating pelvis in XY; keep source metric scale and ground Z.
    shift=np.array([-a['Pelvis'][0],-a['Pelvis'][1],0.])
    a={name:point+shift for name,point in a.items()}
    hands={side:np.array(point)+shift for side,point in source['hand_anchors'].items()}
    length=lambda p,q: float(np.mean([np.linalg.norm(a[s+p]-a[s+q]) for s in ('Left','Right')]))
    thigh=length('Thigh','Shin');shin=length('Shin','Foot')
    upper=length('UpperArm','ForearmHand')
    pelvis_z=float(a['Pelvis'][2])
    waist_z=float(a['Torso'][2]);neck_z=float(a['Head'][2])
    # Copy only the reference manifest data, never mutate the Apollo source.
    refpath=OUT/'apollo_reference.json'
    reference=json.loads(refpath.read_text())
    masses={n:v['mass_kg'] for n,v in reference['bodies'].items()}
    r=E.Element('mujoco',model='wangyu_native_fixed_head_v1')
    E.SubElement(r,'compiler',angle='radian',autolimits='true',fusestatic='false')
    E.SubElement(r,'option',timestep=str(1/120),gravity='0 0 -1.625',integrator='implicitfast',iterations='100')
    default=E.SubElement(r,'default')
    E.SubElement(default,'joint',damping='2',armature='.05',limited='true')
    E.SubElement(default,'geom',contype='0',conaffinity='0',friction='1 .01 .001',rgba='.82 .84 .86 1')
    w=E.SubElement(r,'worldbody');act=E.SubElement(r,'actuator')
    nodes={};positions={};joints=[];geometries=[]
    def body(name,parent,pos):
        pos=np.array(pos);positions[name]=pos
        attrs=dict(name=name,pos=fmt(pos-positions[parent] if parent else pos))
        b=E.SubElement(nodes[parent] if parent else w,'body',**attrs);nodes[name]=b
        if not parent: E.SubElement(b,'freejoint',name='root')
        return b
    def geom(name,body_name,kind,mass,**attrs):
        attrs={k:fmt(v) if not isinstance(v,str) else v for k,v in attrs.items()}
        E.SubElement(nodes[body_name],'geom',name=name,type=kind,mass=str(mass),**attrs)
        geometries.append(dict(name=name,body=body_name,type=kind,mass_kg=mass,**attrs))
    def joint(name,parent,child,axes,limits):
        for axis,(lo,hi) in zip(axes,limits):
            suffix={'X':'Roll','Y':'Pitch','Z':'Yaw'}[axis] if len(axes)>1 else ''
            jname=name+suffix
            axisvec=np.eye(3)['XYZ'.index(axis)]
            E.SubElement(nodes[child],'joint',name=jname,axis=fmt(axisvec),range=fmt(np.deg2rad([lo,hi])))
            torque=180 if any(k in name for k in ('Hip','Knee','Ankle')) else 60
            E.SubElement(act,'motor',name=jname,joint=jname,gear='1',ctrllimited='true',ctrlrange=fmt([-torque,torque]))
            joints.append(dict(name=jname,parent=parent,child=child,axis=axis,range_deg=[lo,hi],effort_limit_nm=torque))
    body('Pelvis',None,a['Pelvis'])
    geom('pelvis','Pelvis','ellipsoid',masses['Pelvis'],size=[.16,.21,.135])
    body('Torso','Pelvis',a['Torso'])
    joint('Waist','Pelvis','Torso',['X','Y'],[[-25,25],[-35,35]])
    chest_h=neck_z-waist_z
    geom('chest','Torso','capsule',masses['Torso']-25,fromto=[0,0,.12,0,0,chest_h-.12],size=[.17])
    geom('backpack','Torso','box',25,pos=[-.26,0,chest_h*.58],size=[.135,.235,.335],rgba='.88 .87 .81 1')
    body('Head','Torso',a['Head'])
    geom('helmet','Head','sphere',masses['Head']-.3,pos=[0,0,.165],size=[.18])
    geom('visor','Head','ellipsoid',.3,pos=[.145,0,.17],size=[.055,.14,.125],rgba='.65 .43 .12 1')
    for side,sign in [('Left',1),('Right',-1)]:
        th=side+'Thigh';sh=side+'Shin';ft=side+'Foot';ua=side+'UpperArm';fa=side+'ForearmHand'
        def capsule_between(name,body_name,end,radius,mass):
            delta=np.asarray(end)-a[body_name]
            unit=delta/np.linalg.norm(delta)
            # Keep rounded caps near the measured joint anchors.
            inset=min(radius*.65,np.linalg.norm(delta)*.2)
            geom(name,body_name,'capsule',mass,fromto=np.r_[unit*inset,delta-unit*inset],size=[radius])
        body(th,'Pelvis',a[th]);joint(side+'Hip','Pelvis',th,['X','Y','Z'],[[-40,40],[-110,45],[-50,50]])
        capsule_between(side.lower()+'_thigh',th,a[sh],.10,masses[th])
        body(sh,th,a[sh]);joint(side+'Knee',th,sh,['Y'],[[0,140]])
        capsule_between(side.lower()+'_shin',sh,a[ft],.08,masses[sh])
        body(ft,sh,a[ft]);joint(side+'Ankle',sh,ft,['Y'],[[-45,45]])
        ankle_z=float(a[ft][2])
        geom(side.lower()+'_sole',ft,'box',masses[ft],pos=[.035,0,-ankle_z/2],size=[.15,.085,ankle_z/2],contype='1',conaffinity='1',rgba='.43 .46 .49 1')
        body(ua,'Torso',a[ua]);joint(side+'Shoulder','Torso',ua,['X','Y','Z'],[[-110,110],[-140,70],[-90,90]])
        capsule_between(side.lower()+'_upper_arm',ua,a[fa],.07,masses[ua])
        body(fa,ua,a[fa]);joint(side+'Elbow',ua,fa,['Y'],[[-150,0]])
        hand_direction=hands[side]-a[fa]
        hand_end=hands[side]+hand_direction/np.linalg.norm(hand_direction)*.09
        capsule_between(side.lower()+'_forearm_hand',fa,hand_end,.06,masses[fa])
    assert len(joints)==20 and len(nodes)==13
    E.indent(r);asset=OUT/'wangyu.xml';E.ElementTree(r).write(asset,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(asset))
    report=dict(version=1,coordinate_convention='+X forward,+Y left,+Z up; right-hand hinge angles',
        joint_parameterization='MuJoCo ordered local X/Y/Z hinges, not USD rotation-vector coordinates. Apollo motions require explicit retargeting.',
        source_fbx_sha256=source['source_sha256'],apollo_reference=str(refpath),apollo_reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
        assumptions=['All joint origins and limb directions preserve the FBX initial posed rig; pelvis centered in XY; source metric scale preserved. Joint q=0 denotes this pose, with hinge ranges relative to it.',
          '126.8 kg total and per-body masses transferred from Apollo engineering envelope estimate, NOT measured Wangyu mass.',
          'Backpack assumed 25 kg included in Torso; primitive inertia recomputed by MuJoCo, NOT copied Apollo inertia.',
          'Primitive radii, torso/helmet/backpack/foot sizes and 60/180 Nm limits are engineering approximations.',
          'Fixed head; hands merged with forearms; only soles collide with ground; no self collision; no exoskeleton.'],
        segment_lengths_m=dict(thigh=thigh,shin=shin,upper_arm=upper),initial_pelvis_height_m=pelvis_z,initial_pose='source_fbx_pose',source_to_physics_translation_m=shift.tolist(),
        body_count=13,actuated_dof_count=20,total_mass_kg=float(model.body_mass.sum()),bodies={},joints=joints,geometries=geometries)
    for name in nodes:
        bid=model.body(name).id
        report['bodies'][name]=dict(neutral_origin_world_m=positions[name].tolist(),mass_kg=float(model.body_mass[bid]),com_local_m=model.body_ipos[bid].tolist(),principal_inertia_kg_m2=model.body_inertia[bid].tolist(),inertial_quat_wxyz=model.body_iquat[bid].tolist())
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    # Standalone scene; MJLab loads wangyu.xml and supplies its own terrain.
    scene=E.Element('mujoco',model='Wangyu native physics preview')
    E.SubElement(scene,'include',file='wangyu.xml')
    visual=E.SubElement(scene,'visual')
    E.SubElement(visual,'global',offwidth='1280',offheight='960')
    E.SubElement(visual,'headlight',ambient='.4 .4 .4',diffuse='.6 .6 .6')
    world=E.SubElement(scene,'worldbody')
    E.SubElement(world,'light',pos='2 -3 4',dir='-1 1 -1')
    E.SubElement(world,'geom',name='ground',type='plane',size='5 5 .1',rgba='.23 .25 .28 1',contype='1',conaffinity='1',friction='1 .01 .001')
    E.indent(scene);E.ElementTree(scene).write(OUT/'scene.xml',encoding='unicode')
    print(f'Built {asset}: 13 bodies, 20 actuators, {report["total_mass_kg"]:.3f} kg')

if __name__=='__main__': build()
