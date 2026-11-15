"""Blender: reduce FBX skin weights to the native reduced physics bodies (20/23 actuated axes).
blender -b --python scripts/export_wangyu_skin.py -- source.fbx
Exports visual-only MuJoCo skins; no physics geoms, mass, or DOFs are added.
"""
import sys
import json
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as E
import bpy
from mathutils import Vector, Quaternion

ROOT=Path(__file__).resolve().parents[1]
source=Path(sys.argv[sys.argv.index('--')+1]).resolve()
out=ROOT/'assets/wangyu_skin';out.mkdir(exist_ok=True)
measure=json.loads((ROOT/'assets/wangyu_physics/rig_measurements.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==measure['source_sha256']
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(source))
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
objects=[o for o in bpy.context.scene.objects if o.type=='MESH' and len(o.data.materials)]
deps=bpy.context.evaluated_depsgraph_get()
low=min((o.matrix_world@v.co).z for o in objects for v in o.evaluated_get(deps).data.vertices)
def xyz(p): return Vector((-p.y,p.x,p.z-low))
def anchor(name): return xyz(rig.matrix_world@rig.pose.bones[name].head)
anchors={n:Vector(p) for n,p in measure['anchors'].items()}
# Physical body frames retain global orientation in the source initial pose.
# Identity bind orientations reproduce source vertices exactly at qpos0.
rot={n:Quaternion((1,0,0,0)) for n in anchors}

def mapped(name):
    if name=='Bones_Pelvis': return 'Pelvis'
    if name in ('Bones_Spine','Bones_Spine1'): return 'Torso'
    if any(k in name.lower() for k in ('neck','head')): return 'Head'
    for side,short in [('Left','L'),('Right','R')]:
        if f'_{short}_' not in name: continue
        if any(k in name for k in ('Forearm','ForeTwist','Hand','Finger')): return side+'ForearmHand'
        if 'UpperArm' in name: return side+'UpperArm'
        if any(k in name for k in ('Foot','Toe')): return side+'Foot'
        if 'Calf' in name: return side+'Shin'
        if 'Thigh' in name: return side+'Thigh'
    # Mechanical/helper bones inherit their nearest physical ancestor.
    bone=rig.data.bones.get(name)
    if bone and bone.parent: return mapped(bone.parent.name)
    return 'Torso'

def fmt(values): return ' '.join(f'{v:.8g}' for v in values)
r=E.Element('mujoco',model='wangyu_visual_skin')
asset=E.SubElement(r,'asset');deform=E.SubElement(r,'deformable')
materials={};report=[]
for obj in objects:
    ev=obj.evaluated_get(deps);mesh=ev.to_mesh();mesh.calc_loop_triangles()
    assert len(mesh.vertices)==len(obj.data.vertices),obj.name
    weights=[]
    for v in obj.data.vertices:
        w=defaultdict(float)
        for g in v.groups:
            if g.weight>1e-6: w[mapped(obj.vertex_groups[g.group].name)]+=g.weight
        if not w:
            if obj.name in ('Glass','Suit_Shield'): w['Head']=1.
            elif obj.parent_type=='BONE': w[mapped(obj.parent_bone)]=1.
            else: w['Torso']=1.
        total=sum(w.values());weights.append({k:v/total for k,v in w.items()})
    for mi in sorted({t.material_index for t in mesh.loop_triangles}):
        mat=mesh.materials[mi];key='visor' if obj.name=='Glass' else mat.name
        if key not in materials:
            name='skin_mat_'+str(len(materials));materials[key]=name
            attrs=dict(name=name,rgba='1 1 1 1',specular='.25',shininess='.25')
            if key=='visor': attrs.update(rgba='.62 .40 .10 1',specular='.9',shininess='.85')
            else:
                texture=source.parent/(mat.name.split('_')[0]+'_BaseColor.png')
                if texture.exists():
                    shutil.copy2(texture,out/texture.name)
                    texname='skin_tex_'+str(len(materials))
                    E.SubElement(asset,'texture',name=texname,type='2d',file='wangyu_skin/'+texture.name,vflip='true')
                    attrs['texture']=texname
            E.SubElement(asset,'material',**attrs)
        vertices=[];uvs=[];faces=[];index={};bones=defaultdict(lambda:([],[]))
        uv=mesh.uv_layers.active
        if uv is None: raise ValueError(f'Missing UV layer: {obj.name}')
        for tri in mesh.loop_triangles:
            if tri.material_index!=mi: continue
            face=[]
            for li in tri.loops:
                vi=mesh.loops[li].vertex_index
                tex=tuple(uv.data[li].uv) if uv else (0.,0.)
                k=(vi,tex)
                if k not in index:
                    idx=len(index);index[k]=idx
                    vertices.extend(xyz(obj.matrix_world@mesh.vertices[vi].co));uvs.extend(tex)
                    for body,weight in weights[vi].items():
                        bones[body][0].append(idx);bones[body][1].append(weight)
                face.append(index[k])
            faces.extend(face)
        skin=E.SubElement(deform,'skin',name='skin_'+obj.name+'_'+str(mi),material=materials[key],group='1',vertex=fmt(vertices),texcoord=fmt(uvs),face=' '.join(map(str,faces)))
        for body,(ids,ws) in bones.items():
            E.SubElement(skin,'bone',body=body,bindpos=fmt(anchors[body]),bindquat=fmt(rot[body]),vertid=' '.join(map(str,ids)),vertweight=fmt(ws))
        report.append(dict(name=obj.name,vertices=len(index),faces=len(faces)//3,bodies=list(bones)))
    ev.to_mesh_clear()
E.indent(r);E.ElementTree(r).write(out/'skin.xml',encoding='unicode')
robot=E.Element('mujoco',model='Wangyu skinned robot')
E.SubElement(robot,'include',file='wangyu_physics/wangyu.xml')
E.SubElement(robot,'include',file='wangyu_skin/skin.xml')
E.indent(robot);E.ElementTree(robot).write(ROOT/'assets/wangyu_skinned_robot.xml',encoding='unicode')
# Includes are resolved against the main model directory: keep the entry in assets/.
scene=E.Element('mujoco',model='Wangyu skinned 20-axis physics')
E.SubElement(scene,'include',file='wangyu_physics/wangyu.xml')
base_scene=E.parse(ROOT/'assets/wangyu_physics/scene.xml').getroot()
for element in base_scene:
    if element.tag!='include': scene.append(element)
E.SubElement(scene,'include',file='wangyu_skin/skin.xml')
E.indent(scene);E.ElementTree(scene).write(ROOT/'assets/wangyu_skinned.xml',encoding='unicode')
(out/'binding.json').write_text(json.dumps(dict(source=str(source),source_sha256=measure['source_sha256'],method='Native MuJoCo linear blend skinning driven by actual physics bodies; FBX weights pooled over reduced rigid segments; source initial posed body frames preserved, with identity bind orientations.',texture_coordinates='Source Blender UVs preserved; MuJoCo PNG textures use vflip=true to match Blender image origin.',head_and_wrists='No added joints: head follows fixed Head body; all hand/finger weights pooled into forearm.',meshes=report,bind_positions={k:list(v) for k,v in anchors.items()},bind_quaternions_wxyz={k:list(v) for k,v in rot.items()}),indent=2)+'\n')
print('Exported',len(report),'native MuJoCo skins to',out/'skin.xml')
