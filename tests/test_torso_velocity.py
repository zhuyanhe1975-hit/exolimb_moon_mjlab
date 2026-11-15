"""Static native kinematic velocity queries only; no forward dynamics or stepping."""
from types import SimpleNamespace
import warnings
import unittest
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
import flat_support_task as flat
import torso_velocity


class TorsoVelocity(unittest.TestCase):
    def test_angular_penalty_covers_roll_pitch_yaw(self):
        class Model:
            def sensor(self,name):return SimpleNamespace(dim=[3],adr=[0])
        values=torch.tensor([[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
        env=SimpleNamespace(sim=SimpleNamespace(mj_model=Model(),data=SimpleNamespace(sensordata=values)))
        torch.testing.assert_close(torso_velocity.angular_squared(env),torch.tensor([1.,4.,9.]))

    def test_waist_rotation_moves_torso_com_without_pelvis_translation(self):
        cfg=flat.terrain_stand_cfg();cfg.scene.num_envs=1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');scene=Scene(cfg.scene,'cpu');model=scene.compile()
        data=mujoco.MjData(model)
        joint=model.joint('robot/WaistPitch').id
        data.qvel[int(model.jnt_dofadr[joint])]=.4
        mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        mujoco.mj_comVel(model,data);mujoco.mj_sensorVel(model,data)
        sensor=model.sensor('robot/'+torso_velocity.SENSOR_NAME)
        self.assertEqual(int(sensor.objtype[0]),int(mujoco.mjtObj.mjOBJ_BODY))
        env=SimpleNamespace(sim=SimpleNamespace(mj_model=model,
            data=SimpleNamespace(sensordata=torch.tensor(data.sensordata)[None])))
        velocity=torso_velocity.velocity(env)[0].numpy()
        jac=np.zeros((3,model.nv))
        mujoco.mj_jacBodyCom(model,data,jac,None,model.body('robot/Torso').id)
        np.testing.assert_allclose(velocity,jac@data.qvel,atol=1e-12)
        jac_angular=np.zeros((3,model.nv))
        mujoco.mj_jacBodyCom(model,data,None,jac_angular,model.body('robot/Torso').id)
        np.testing.assert_allclose(torso_velocity.angular_velocity(env)[0].numpy(),jac_angular@data.qvel,atol=1e-12)
        self.assertAlmostEqual(float(torso_velocity.angular_squared(env)[0]),.16,places=10)
        self.assertEqual(cfg.rewards['torso_angular_velocity'].weight,-.2)
        root=model.sensor('robot/balance_root_vel');start=int(root.adr[0])
        np.testing.assert_allclose(data.sensordata[start:start+3],0,atol=1e-12)
        self.assertGreater(np.linalg.norm(velocity),.01)
        self.assertGreater(float(torso_velocity.squared(env)[0]),0)
        self.assertEqual(data.time,0.)
        self.assertNotIn('pelvis_velocity',cfg.rewards)
        self.assertEqual(cfg.rewards['torso_velocity'].weight,-4.)
        self.assertEqual(flat.OBS_DIM,151)


if __name__=='__main__':unittest.main()
