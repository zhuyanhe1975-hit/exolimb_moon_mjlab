"""Blender static source-material/UV audit and CPU reference render; no physics."""
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
info=json.loads((ROOT/'assets/wangyu_skin/binding.json').read_text())
source=Path(info['source'])
out=ROOT/'outputs/skin_audit';out.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(source))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and len(o.data.materials)]
report={'source':str(source),'meshes':[]}
for obj in meshes:
    item={'name':obj.name,'vertices':len(obj.data.vertices),'uv_layers':[u.name for u in obj.data.uv_layers],
          'active_uv':obj.data.uv_layers.active.name if obj.data.uv_layers.active else None,'materials':[],
          'armature_modifiers':[{'object':mod.object.name if mod.object else None,'preserve_volume':mod.use_deform_preserve_volume}
                                for mod in obj.modifiers if mod.type=='ARMATURE']}
    for material in obj.data.materials:
        entry={'name':material.name,'nodes':[],'links':[]}
        if material.use_nodes:
            for node in material.node_tree.nodes:
                n={'name':node.name,'type':node.type}
                if node.type=='TEX_IMAGE' and node.image:
                    image=node.image
                    candidate=source.parent/Path(image.filepath.replace('\\','/')).name
                    if candidate.exists():image.filepath=str(candidate);image.reload()
                    n.update(image=image.name,file=image.filepath,exists=Path(bpy.path.abspath(image.filepath)).is_file(),
                             extension=node.extension,projection=node.projection)
                if node.type=='UVMAP':n['uv_map']=node.uv_map
                entry['nodes'].append(n)
            entry['links']=[{'from':l.from_node.name,'from_socket':l.from_socket.name,
                             'to':l.to_node.name,'to_socket':l.to_socket.name} for l in material.node_tree.links]
        item['materials'].append(entry)
    report['meshes'].append(item)
(out/'source_materials.json').write_text(json.dumps(report,indent=2)+'\n')
if '--render' not in sys.argv:raise SystemExit(0)
# FBX's VRay materials import without image nodes in Blender. Restore the
# supplied material-name BaseColor assignment explicitly for UV reference QA.
for material in bpy.data.materials:
    path=source.parent/(material.name.split('_')[0]+'_BaseColor.png')
    if path.exists() and material.use_nodes:
        shader=next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        texture=material.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image=bpy.data.images.load(str(path),check_existing=True)
        material.node_tree.links.new(texture.outputs['Color'],shader.inputs['Base Color'])
deps=bpy.context.evaluated_depsgraph_get()
points=[]
for obj in meshes:
    ev=obj.evaluated_get(deps)
    points.extend(obj.matrix_world@Vector(c) for c in ev.bound_box)
low=Vector([min(p[i] for p in points) for i in range(3)])
high=Vector([max(p[i] for p in points) for i in range(3)])
center=(low+high)/2;height=high.z-low.z
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=900;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new('AuditWorld');scene.world.color=(.3,.3,.3)
bpy.ops.object.camera_add(location=center+Vector((0.,-4.,1.0))*height/1.8)
camera=bpy.context.object;camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.type='ORTHO';camera.data.ortho_scale=height*1.15;scene.camera=camera
for offset,power,size in [((2,-4,5),500,4),((-3,-1,2),220,3),((1,3,4),350,3)]:
    bpy.ops.object.light_add(type='AREA',location=center+Vector(offset))
    light=bpy.context.object;light.data.energy=power;light.data.shape='DISK';light.data.size=size
    light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
scene.render.image_settings.file_format='PNG'
scene.render.filepath=str(out/'blender_source.png')
bpy.ops.render.render(write_still=True)
