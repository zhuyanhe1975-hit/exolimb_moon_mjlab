"""Compile-only physical invariants; no mj_forward/mj_step or GPU work."""
import unittest
import numpy as np
import mujoco
import balance_task
import tiled_support_task as tiled


class TiledSupport(unittest.TestCase):
    def test_exact_bounds_native_inertia_and_actuators_preserved(self):
        source=balance_task.robot_spec().compile()
        model=tiled.robot_spec().compile()
        self.assertEqual((source.nq,source.nv,source.nu),(model.nq,model.nv,model.nu))
        for field in ('body_mass','body_ipos','body_iquat','body_inertia','jnt_range',
                      'actuator_gainprm','actuator_biasprm','actuator_forcerange','site_pos','site_size'):
            np.testing.assert_allclose(getattr(model,field),getattr(source,field),atol=1e-12,rtol=1e-12)
        for side in ('left','right'):
            original=source.geom(side+'_sole');rotation=np.zeros(9)
            mujoco.mju_quat2Mat(rotation,original.quat);rotation=rotation.reshape(3,3)
            centers=[]
            for index in range(4):
                geom=model.geom(side+'_sole'+(f'_tile_{index}' if index else ''))
                self.assertEqual(int(geom.contype[0]),int(original.contype[0]))
                self.assertEqual(int(geom.conaffinity[0]),int(original.conaffinity[0]))
                np.testing.assert_allclose(geom.friction,original.friction)
                center=rotation.T@(geom.pos-original.pos);centers.append(center)
                np.testing.assert_allclose(geom.size,original.size*np.array([.5,.5,1.]))
                self.assertTrue((abs(center)+geom.size<=original.size+1e-12).all())
            np.testing.assert_allclose(np.mean(centers,axis=0),0,atol=1e-12)
            self.assertEqual(len(set(tuple(np.round(x,10)) for x in centers)),4)
        self.assertEqual(model.ngeom,source.ngeom+6)

    def test_actual_skin_attachment_compiles_with_same_physics(self):
        parent=mujoco.MjSpec()
        skin=tiled.skin_spec()
        parent.option.timestep=skin.option.timestep
        parent.option.integrator=skin.option.integrator
        parent.option.gravity=skin.option.gravity
        parent.attach(skin,prefix='robot/',frame=parent.worldbody.add_frame())
        model=parent.compile();physical=tiled.robot_spec().compile()
        self.assertEqual(model.nskin,9)
        np.testing.assert_allclose(model.body_mass,physical.body_mass,atol=1e-12)
        np.testing.assert_allclose(model.body_inertia,physical.body_inertia,atol=1e-12)
        np.testing.assert_allclose(model.geom_size,physical.geom_size,atol=1e-12)

    def test_native_ccd_profile_survives_mjlab_option_application(self):
        cfg=tiled.terrain_stand_cfg()
        parent=mujoco.MjSpec()
        child=tiled.robot_spec()
        parent.option.timestep=child.option.timestep
        parent.option.integrator=child.option.integrator
        parent.option.gravity=child.option.gravity
        parent.attach(child,prefix='robot/',frame=parent.worldbody.add_frame())
        cfg.scene.spec_fn(parent)
        model=parent.compile()
        cfg.sim.mujoco.apply(model)
        self.assertEqual(model.opt.ccd_tolerance,.001)
        self.assertIn('_ccd1e3',tiled.EXPERIMENT)

    def test_contact_sensor_aggregates_tiles_per_foot(self):
        cfg=tiled.terrain_stand_cfg()
        sensor=next(s for s in cfg.scene.sensors if s.name=='support_contacts')
        self.assertEqual(sensor.primary.mode,'body')
        self.assertEqual(sensor.primary.pattern,('LeftFoot','RightFoot'))
        self.assertEqual(sensor.num_slots,4)
        self.assertEqual(cfg.scene.num_envs,8192)
        self.assertEqual(cfg.rewards['joint_pose'].weight,-.25)
        self.assertEqual(cfg.rewards['centroidal_angular_momentum'].weight,-.002)


if __name__=='__main__':unittest.main()
