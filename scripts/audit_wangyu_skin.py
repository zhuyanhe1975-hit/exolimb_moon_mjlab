"""Static 23-axis skin/UV/material QA; never advances physics or runs a policy.

MUJOCO_GL=egl <shared-python> scripts/audit_wangyu_skin.py
Uses the same robot/ attachment and material-name handling as actual playback.
"""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as E

import mujoco
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from balance_task import robot_spec,skin_spec

OUT=ROOT/'outputs/skin_audit';OUT.mkdir(parents=True,exist_ok=True)


def scene(spec):
    parent=mujoco.MjSpec()
    parent.option.timestep=1/120
    parent.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    parent.attach(spec,prefix='robot/',frame=parent.worldbody.add_frame())
    parent.worldbody.add_geom(name='ground',type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[5,5,.1],rgba=[.18,.19,.21,1])
    model=parent.compile()
    model.vis.global_.offheight=1000;model.vis.global_.offwidth=1000
    model.vis.headlight.ambient=[.22,.22,.22]
    model.vis.headlight.diffuse=[.5,.5,.5]
    return model


def main():
    base=scene(robot_spec());model=scene(skin_spec())
    assert (model.nu,model.nv,model.nq,model.nskin)==(23,29,30,9)
    for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','jnt_range',
                  'jnt_axis','jnt_pos','qpos0','dof_damping','dof_armature','actuator_gainprm',
                  'actuator_biasprm','actuator_forcerange','geom_size','geom_pos','geom_type',
                  'geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp'):
        np.testing.assert_array_equal(getattr(base,field),getattr(model,field))
    skin_xml=E.parse(ROOT/'assets/wangyu_skin/skin.xml').getroot()
    weight_error=0.
    for skin in skin_xml.findall('deformable/skin'):
        count=len(np.fromstring(skin.get('vertex'),sep=' '))//3
        total=np.zeros(count)
        for bone in skin.findall('bone'):
            ids=np.fromstring(bone.get('vertid'),sep=' ',dtype=int)
            weights=np.fromstring(bone.get('vertweight'),sep=' ')
            assert len(ids)==len(weights) and (weights>=0).all()
            np.add.at(total,ids,weights)
        weight_error=max(weight_error,float(np.max(np.abs(total-1.))))
        np.testing.assert_allclose(total,1.,atol=2e-7,rtol=0)
    textures={t.get('name'):t for t in skin_xml.findall('asset/texture')}
    for name,texture in textures.items():
        assert texture.get('vflip')=='true'
        image=np.array(Image.open(ROOT/'assets'/texture.get('file')).convert('RGB'))
        tid=model.texture('robot/'+name).id
        h,w=int(model.tex_height[tid]),int(model.tex_width[tid]);start=int(model.tex_adr[tid])
        actual=model.tex_data[start:start+h*w*3].reshape(h,w,3)
        np.testing.assert_array_equal(actual,image[::-1])
    for skin in range(model.nskin):
        assert model.skin_matid[skin]>=0
        name=model.material(int(model.skin_matid[skin])).name
        assert name.startswith('robot/skin_mat_')
    option=mujoco.MjvOption();option.geomgroup[3]=False
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.95];cam.distance=3.;cam.elevation=-10
    data=mujoco.MjData(model)
    with mujoco.Renderer(model,height=900,width=800) as renderer:
        mujoco.mj_forward(model,data)
        cam.azimuth=180
        renderer.update_scene(data,camera=cam,scene_option=option)
        neutral=renderer.scene.skinvert.copy().reshape(-1,3)
        shift=json.loads((ROOT/'assets/wangyu_physics/manifest.json').read_text())['source_to_physics_translation_m']
        source_error=float(np.max(np.abs(neutral-(model.skin_vert+shift))))
        assert source_error<1e-6
        Image.fromarray(renderer.render()).save(OUT/'fixed_front.png')
        cam.azimuth=0
        renderer.update_scene(data,camera=cam,scene_option=option)
        Image.fromarray(renderer.render()).save(OUT/'fixed_back.png')
        displacement={}
        for jid in range(1,model.njnt):
            mujoco.mj_resetData(model,data)
            data.qpos[model.jnt_qposadr[jid]]=.15 if model.jnt_range[jid,1]>=.15 else -.15
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=cam,scene_option=option)
            moved=renderer.scene.skinvert.reshape(-1,3)
            assert np.isfinite(moved).all()
            delta=float(np.max(np.linalg.norm(moved-neutral,axis=1)))
            assert delta>.001,model.joint(jid).name
            displacement[model.joint(jid).name]=delta
        mujoco.mj_resetData(model,data)
        for side in ('Left','Right'):
            for suffix,value in (('HipPitch',-.15),('Knee',.35),('Ankle',-.15),('Elbow',-.45)):
                data.qpos[model.joint('robot/'+side+suffix).qposadr[0]]=value
        mujoco.mj_forward(model,data)
        cam.azimuth=135
        renderer.update_scene(data,camera=cam,scene_option=option)
        Image.fromarray(renderer.render()).save(OUT/'fixed_bend.png')
    report={'actuated_axes':23,'skins':model.nskin,'vertices':model.nskinvert,
            'physical_parameters_unchanged':True,'robot_namespace_materials_valid':True,
            'texture_vflip_verified_against_png_bytes':True,'max_weight_sum_error':weight_error,
            'source_pose_max_vertex_error_m':source_error,'joint_displacement_m':displacement,
            'physics_steps':0,'scope':'Static poses and rendering only; not a policy or dynamic deformation test.'}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
