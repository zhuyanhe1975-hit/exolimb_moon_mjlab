"""Blender: extract posed rig anchors for native primitive fitting from Wangyu FBX.
Usage: blender -b --python scripts/extract_wangyu_physics.py -- source.fbx
"""
import sys,json,hashlib
from pathlib import Path
import bpy

source=Path(sys.argv[sys.argv.index('--')+1]).resolve()
out=Path(__file__).resolve().parents[1]/'assets/wangyu_physics'
out.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(source))
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
# Exclude FBX rig-control meshes (no material); include suit, hardware and helmet.
objects=[o for o in bpy.context.scene.objects if o.type=='MESH' and len(o.data.materials)]
deps=bpy.context.evaluated_depsgraph_get()
points=[o.matrix_world@v.co for o in objects for v in o.evaluated_get(deps).data.vertices]
low=min(p.z for p in points)
# Preserve source metric scale, rotate -Y forward into +X, X left into +Y.
def xyz(p): return [-float(p.y),float(p.x),float(p.z-low)]
def anchor(name): return xyz(rig.matrix_world@rig.pose.bones[name].head)
anchors={'Pelvis':anchor('Bones_Pelvis'),'Torso':anchor('Bones_Spine'),'Head':anchor('Bones_Neck')}
for side,short in [('Left','L'),('Right','R')]:
 for part,bone in [('Thigh','Thigh'),('Shin','Calf'),('Foot','Foot'),('UpperArm','UpperArm'),('ForearmHand','Forearm')]:
  anchors[side+part]=anchor('Bones_'+short+'_'+bone)

hands={side:anchor('Bones_'+short+'_Hand') for side,short in [('Left','L'),('Right','R')]}
report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),anchors=anchors,hand_anchors=hands,coordinate_convention='+X forward,+Y left,+Z up',height_m=max(p.z for p in points)-low,note='Posed source rig, preserved as native qpos0. Not calibrated anthropometry.')
(out/'rig_measurements.json').write_text(json.dumps(report,indent=2)+'\n')
print('Extracted rig measurements:',out/'rig_measurements.json')
