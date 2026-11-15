"""Static transfer contracts. No simulation step or Kit launch."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sim2sim_contract import arguments,height,timing,resolve_terrain_mode
from sim2sim_sensors import NativeSensors
from scripts.sim2sim_isaacsim import enable_foot_contact_reporting,validate_foot_contact_views
from scripts import sim2sim_isaacsim as entry
from sim2sim_instances import prepare_instances,sample_root


class TransferContract(unittest.TestCase):
    def test_sixteen_humans_share_one_patch_with_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(num_envs=16,patch=0,seed=42,yaw=0,bundle=Path(directory))
            bank={'normalized':np.zeros((1,161,161)),'height_range':np.ones(1),'zmin':np.zeros(1)}
            manifest={'terrain':{'mode':'lunar'}}
            points=np.array([[-.15,-.2,-.8],[.15,-.2,-.8],[-.15,.2,-.8],[.15,.2,-.8]])
            prepare_instances(args,manifest,bank,points,.01)
            data=np.load(Path(directory)/'instances.npz');positions=data['roots'][:,:2]
            distance=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1);np.fill_diagonal(distance,np.inf)
            self.assertGreaterEqual(distance.min(),1.5)
            np.testing.assert_array_equal(data['patches'],0);np.testing.assert_array_equal(data['offsets'],0)
            self.assertLess(abs(positions).max(),5.6)
            self.assertGreater(np.ptp(data['roots'][:,6]),.5)
            root=sample_root(data['grids'][0],data['spawn_xy'][0],points,data['padding'][0],np.random.default_rng(7),
                             limit=7.,excluded=data['roots'][1:])
            self.assertGreaterEqual(np.linalg.norm(data['roots'][1:,:2]-root[:2],axis=-1).min(),1.5)

    def test_failure_printed_before_kit_close_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle=Path(directory);(bundle/'manifest.json').write_text(json.dumps({'checkpoint':'model_200.pt'}))
            args=SimpleNamespace(bundle=bundle,prepare_only=False,verbose_init=True,headless=True,report=None)
            app=Mock();output=io.StringIO();errors=io.StringIO()
            def close(*,exit_code):
                self.assertIn('ValueError: unavailable contact backend',errors.getvalue())
                failure=json.loads((bundle/'report.failure.json').read_text())
                self.assertIn('ValueError: unavailable contact backend',failure['traceback'])
                self.assertEqual(exit_code,1)
            app.close.side_effect=close
            fake_isaac=SimpleNamespace(SimulationApp=Mock(return_value=app))
            with patch.object(entry,'arguments',return_value=args),patch.object(entry,'run',side_effect=ValueError('unavailable contact backend')),patch.dict(sys.modules,{'isaacsim':fake_isaac}),contextlib.redirect_stdout(output),contextlib.redirect_stderr(errors):
                with self.assertRaisesRegex(ValueError,'unavailable contact backend'):entry.main()
            app.close.assert_called_once_with(exit_code=1)

    def test_native_contact_reporting_before_view_binding(self):
        class NativeFoot:
            def __init__(self,side):
                self.paths=[f'/World/Robot/{side}Foot'];self.reporting=False;self.bound=False
            def set_enabled_contact_tracking(self,enabled,*,threshold):
                self.assert_before_binding=not self.bound
                self.reporting=enabled;self.threshold=threshold
            def get_enabled_contact_tracking(self):return np.array([[self.reporting]])
            def is_physics_tensor_entity_valid(self):return self.bound
            @property
            def num_contact_filters(self):
                if not self.reporting:raise AttributeError("'NoneType' object has no attribute 'filter_count'")
                return 1
            def get_contact_force_data(self,*,dt):
                self.dt=dt
                return (np.zeros((64,1)),np.zeros((64,3)),np.zeros((64,3)),np.zeros((64,1)),np.zeros((1,1),dtype=int),np.zeros((1,1),dtype=int))
        feet=[NativeFoot(side) for side in ('Left','Right')]
        enable_foot_contact_reporting(feet)
        for foot in feet:
            self.assertTrue(foot.assert_before_binding);self.assertTrue(foot.reporting)
            self.assertEqual(foot.threshold,0.);foot.bound=True
        validate_foot_contact_views(feet,1/120)
        for foot in feet:self.assertEqual(foot.dt,1/120)
        feet[1].reporting=False
        with self.assertRaisesRegex(RuntimeError,'RightFoot') as caught:
            validate_foot_contact_views(feet,1/120)
        self.assertIsInstance(caught.exception.__cause__,AttributeError)

    def test_explicit_current_terrain_transfer(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            arguments(['--terrain','current','--strict-terrain'])
        self.assertTrue(arguments(['--terrain','current']).terrain_transfer)
        self.assertEqual(arguments([]).terrain,'auto')
        self.assertEqual(arguments([]).physics_substeps,4)
        self.assertTrue(arguments(['--terrain','moon']).terrain_transfer)
        self.assertEqual(resolve_terrain_mode('moon','stand-moon',Path('/unused/model.pt'),True)[0],'current')
        self.assertEqual(arguments(['--terrain','moon','--num-envs','16']).num_envs,16)
        self.assertEqual(arguments(['--terrain','moon','--num_envs','16']).num_envs,16)
        self.assertEqual(arguments([]).render_every,1)
        self.assertEqual(arguments(['--num-envs','16']).render_every,2)
        self.assertEqual(arguments(['--num-envs','16','--render-every','1']).render_every,1)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):arguments(['--render-every','0'])
        for value in ('0','65'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):arguments(['--num-envs',value])

    def test_auto_terrain_saved_missing_and_corrupt(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'model_1.pt'
            selected,reason=resolve_terrain_mode('auto','stand-moon',checkpoint,True)
            self.assertEqual(selected,'current');self.assertIn('not exact replay',reason)
            with self.assertRaises(ValueError):resolve_terrain_mode('auto','stand-moon',checkpoint,False)
            folder=Path(directory)/'terrain';folder.mkdir();archive=folder/'bank.npz';archive.write_bytes(b'saved bank')
            (folder/'metadata.json').write_text(json.dumps({'physics_revision':'native_groundsync1',
                'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'fingerprint':'snapshot fingerprint'}))
            self.assertEqual(resolve_terrain_mode('auto','stand-moon',checkpoint,True)[0],'trained')
            archive.write_bytes(b'changed bank')
            self.assertEqual(resolve_terrain_mode('auto','stand-moon',checkpoint,True)[0],'current')
            with self.assertRaises(ValueError):resolve_terrain_mode('auto','stand-moon',checkpoint,False)

    def test_substep_timing_preserves_policy_period(self):
        for subdivisions in (1,2,4,8):
            dt,decimation,period=timing({'physics_dt':1/120,'decimation':2},subdivisions)
            self.assertAlmostEqual(period,1/60);self.assertAlmostEqual(dt*decimation,period)
            self.assertEqual(decimation,2*subdivisions)
        self.assertAlmostEqual(timing({'physics_dt':1/120,'decimation':2},4)[0],1/480)
        with self.assertRaises(ValueError):timing({'physics_dt':1/120,'decimation':2},0)

    def test_finite_duration_and_heading(self):
        for args in (['--seconds','0'],['--episode-seconds','-1'],['--yaw','nan']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):arguments(args)

    def test_triangle_height_against_native_ray(self):
        spec=mujoco.MjSpec();field=spec.add_hfield(name='ground',nrow=4,ncol=4,size=[2.,2.,1.,1.])
        values=np.array([[0,.1,.7,.2],[.1,.4,.2,.5],[.7,.3,.9,.1],[.3,.4,.5,1.]])
        field.userdata=values.ravel().tolist()
        spec.worldbody.add_geom(name='ground',type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname='ground',group=0)
        model=spec.compile();data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
        xy=np.random.default_rng(92).uniform(-1.99,1.99,(100,2))
        expected=[]
        for x,y in xy:
            distance=mujoco.mj_ray(model,data,np.array([x,y,3.]),np.array([0.,0.,-1.]),None,True,-1,None)
            self.assertGreater(distance,0);expected.append(3-distance)
        np.testing.assert_allclose(height(xy,values,2.),expected,atol=2e-7)
        with self.assertRaises(ValueError):height(np.array([[2.1,0.]]),values,2.)

    def test_touch_volume_uses_rotated_site_and_normal_load(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="1 2 3" euler=".3 .5 .8">
          <site name="balance_left_sole" type="box" pos=".2 0 -.3" size=".15 .085 .015"/>
        </body></worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
        reader=NativeSensors.__new__(NativeSensors);reader.model=model;reader.data=data
        local=np.array([[0,0,0],[.1,.04,.01],[0,0,.03]])
        world=local@data.site_xmat[0].reshape(3,3).T+data.site_xpos[0]
        self.assertAlmostEqual(reader.touch_load('left',np.array([[80.],[-20.],[900.]]),world),100.)


if __name__=='__main__':unittest.main()
