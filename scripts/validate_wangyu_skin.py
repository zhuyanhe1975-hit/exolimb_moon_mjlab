"""Check visual skin deformation and unchanged native physics; render QA poses."""
from pathlib import Path
import json
import numpy as np
import mujoco
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/wangyu_physics'
base=mujoco.MjModel.from_xml_path(str(ROOT/'assets/wangyu_physics/scene.xml'))
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/wangyu_skinned.xml'))
assert m.nu==base.nu==20 and m.nv==base.nv==26 and m.nbody==base.nbody
assert m.nskin==9 and m.nflex==0 and m.nmesh==0
for name in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','jnt_range','dof_damping','dof_armature','actuator_ctrlrange','geom_size','geom_pos','geom_type','geom_contype','geom_conaffinity','geom_friction'):
    np.testing.assert_allclose(getattr(m,name),getattr(base,name),rtol=0,atol=0)
d=mujoco.MjData(m);b=mujoco.MjData(base)
# A deterministic 2s physics comparison, with visual-only assets on one side.
for step in range(240):
    control=3*np.sin(step*.05+np.arange(m.nu))
    b.ctrl[:]=control;d.ctrl[:]=control
    mujoco.mj_step(base,b);mujoco.mj_step(m,d)
    np.testing.assert_allclose(d.qpos,b.qpos,rtol=0,atol=1e-12)
    assert np.isfinite(d.qpos).all()
option=mujoco.MjvOption();option.geomgroup[3]=False
for i in range(m.ngeom):
    if m.geom_bodyid[i]!=0: m.geom_group[i]=3
cam=mujoco.MjvCamera();cam.lookat[:]=(0,0,.95);cam.distance=3.5;cam.azimuth=135;cam.elevation=-12
with mujoco.Renderer(m,640,800) as renderer:
    mujoco.mj_resetData(m,d);mujoco.mj_forward(m,d)
    renderer.update_scene(d,camera=cam,scene_option=option)
    neutral=renderer.scene.skinvert.copy()
    assert np.isfinite(neutral).all()
    # Source FBX coordinates are preserved at qpos0, with only XY centering.
    manifest=json.loads((ROOT/'assets/wangyu_physics/manifest.json').read_text())
    shift=np.array(manifest['source_to_physics_translation_m'])
    expected=m.skin_vert+shift
    source_pose_error=float(np.max(np.abs(neutral.reshape(-1,3)-expected)))
    assert source_pose_error<1e-6
    measurements=json.loads((ROOT/'assets/wangyu_physics/rig_measurements.json').read_text())
    for name,point in measurements['anchors'].items():
        np.testing.assert_allclose(d.xpos[m.body(name).id],np.array(point)+shift,atol=1e-8,rtol=0)
    Image.fromarray(renderer.render()).save(OUT/'skinned_preview.png')
    # Each driven hinge must change the rendered suit, not just a hidden proxy.
    displacement={}
    for jid in range(1,m.njnt):
        mujoco.mj_resetData(m,d)
        angle=.25 if m.jnt_range[jid,1]>=.25 else -.25
        d.qpos[m.jnt_qposadr[jid]]=angle
        mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam,scene_option=option)
        moved=renderer.scene.skinvert
        assert np.isfinite(moved).all()
        delta=float(np.max(np.linalg.norm((moved-neutral).reshape(-1,3),axis=1)))
        assert delta>.001,m.joint(jid).name
        displacement[m.joint(jid).name]=delta
    mujoco.mj_resetData(m,d)
    for side in ('Left','Right'):
        for joint,angle in [('HipPitch',-.35),('Knee',.7),('Ankle',-.35),('ShoulderPitch',-.2),('Elbow',-.6)]:
            d.qpos[m.joint(side+joint).qposadr[0]]=angle
    mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam,scene_option=option)
    Image.fromarray(renderer.render()).save(OUT/'skinned_bend.png')
report=dict(skins=m.nskin,vertices=m.nskinvert,actuators=m.nu,head_wrists_fixed=True,initial_pose='source_fbx_pose',source_pose_max_vertex_error_m=source_pose_error,physical_parameters_identical=True,physics_comparison_steps=240,physics_max_qpos_tolerance=1e-12,all_20_axes_deform_skin=True,max_vertex_displacement_by_joint_m=displacement)
(OUT/'skin_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
