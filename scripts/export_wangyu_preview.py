"""Run in Blender to bake the source FBX into static MuJoCo visual meshes."""
import sys
import shutil
from pathlib import Path
import xml.etree.ElementTree as E
import bpy
from mathutils import Vector

source = Path(sys.argv[sys.argv.index('--') + 1])
out = Path(__file__).resolve().parents[1] / 'assets/wangyu_visual'
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(source))
objects = [o for o in bpy.context.scene.objects if o.type == 'MESH' and not o.hide_render]
points = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
lo = Vector([min(p[i] for p in points) for i in range(3)])
hi = Vector([max(p[i] for p in points) for i in range(3)])
center = Vector(((lo.x+hi.x)/2, (lo.y+hi.y)/2, lo.z))
scale = 1.8 / (hi.z-lo.z)
r = E.Element('mujoco', model='Wangyu visual preview')
E.SubElement(r, 'compiler', angle='radian')
a = E.SubElement(r, 'asset')
w = E.SubElement(r, 'worldbody')
E.SubElement(w, 'light', pos='2 -3 4', dir='-1 1 -1')
E.SubElement(w, 'geom', type='plane', size='5 5 .1', rgba='.18 .20 .23 1')
body = E.SubElement(w, 'body', name='wangyu')
deps = bpy.context.evaluated_depsgraph_get()
for oi, obj in enumerate(objects):
    evaluated = obj.evaluated_get(deps)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    for mi in sorted({t.material_index for t in mesh.loop_triangles}):
        name = f'part_{oi}_{mi}'
        mat = mesh.materials[mi] if mi < len(mesh.materials) else None
        rgba = '0.85 0.86 0.88 1'
        attrs = dict(name=name, rgba=rgba, specular='.3', shininess='.3')
        if mat:
            texture = source.parent / (mat.name.split('_')[0] + '_BaseColor.png')
            if texture.exists():
                shutil.copy2(texture, out / texture.name)
                E.SubElement(a, 'texture', name=name, type='2d', file=texture.name, vflip='true')
                attrs.update(texture=name, rgba='1 1 1 1')
        if mat and mat.use_nodes:
            images = [n.image for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image and 'basecolor' in n.image.name.lower()]
            if images:
                original = source.parent / Path(images[0].filepath.replace('\\', '/')).name
                if original.exists():
                    dest = out / original.name
                    if not dest.exists(): shutil.copy2(original, dest)
                    E.SubElement(a, 'texture', name=name, type='2d', file=dest.name, vflip='true')
                    attrs['texture'] = name
                    attrs['rgba'] = '1 1 1 1'
        if 'glass' in obj.name.lower():
            attrs.update(rgba='.55 .32 .08 1', specular='.9', shininess='.9')
        E.SubElement(a, 'material', **attrs)
        lines = []
        uv = mesh.uv_layers.active
        idx = 1
        for tri in mesh.loop_triangles:
            if tri.material_index != mi: continue
            for li in tri.loops:
                p = (obj.matrix_world @ mesh.vertices[mesh.loops[li].vertex_index].co - center) * scale
                lines.append(f'v {p.x} {p.y} {p.z}\n')
                t = uv.data[li].uv if uv else (0, 0)
                lines.append(f'vt {t[0]} {t[1]}\n')
                n = obj.matrix_world.to_3x3().inverted().transposed() @ mesh.corner_normals[li].vector
                n.normalize()
                lines.append(f'vn {n.x} {n.y} {n.z}\n')
            lines.append(f'f {idx}/{idx}/{idx} {idx+1}/{idx+1}/{idx+1} {idx+2}/{idx+2}/{idx+2}\n')
            idx += 3
        (out / f'{name}.obj').write_text(''.join(lines))
        E.SubElement(a, 'mesh', name=name, file=f'{name}.obj')
        E.SubElement(body, 'geom', type='mesh', mesh=name, material=name, contype='0', conaffinity='0')
    evaluated.to_mesh_clear()
E.indent(r)
E.ElementTree(r).write(out / 'scene.xml', encoding='unicode')
print('Exported', out / 'scene.xml')
