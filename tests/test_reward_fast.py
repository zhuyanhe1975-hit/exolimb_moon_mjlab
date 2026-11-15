"""CPU tensor equivalence and phase/reset lifecycle; no physics stepping."""
import os
from types import SimpleNamespace
from unittest.mock import patch
import torch
import balance_mdp as mdp
from lunar_terrain import Surface


def test_scan_reuses_same_phase_center_exactly():
    surface=Surface.__new__(Surface)
    axis=torch.linspace(-.5,.5,5)
    surface.scan_points=torch.cartesian_prod(axis,axis)
    calls=[]
    def height(xy):
        calls.append(xy.shape)
        return .2*xy[...,0]-.1*xy[...,1]+.03*xy[...,0].square()
    surface.height=height
    root=torch.tensor([[1.,2.,3.,1.,0.,0.,0.],[-2.,1.,4.,.8,0.,0.,.6]])
    expected=surface.scan(root)
    center=height(root[:,:2]);calls.clear()
    torch.testing.assert_close(surface.scan(root,center_height=center),expected,rtol=0,atol=0)
    assert calls==[torch.Size([2,25,2])]


def test_reward_only_matches_all_legacy_columns():
    generator=torch.Generator().manual_seed(1729)
    for dtype in (torch.float32,torch.float64):
        x=torch.randn(257,mdp.STATE_DIM+mdp.SENSOR_DIM+30,generator=generator,dtype=dtype)
        offset=mdp.STATE_DIM
        for name,_,_,_,width in mdp.SENSORS:
            if name.endswith('quat'):
                x[:,offset:offset+width]=torch.nn.functional.normalize(x[:,offset:offset+width],dim=-1)
            if name.endswith('load'):x[:30,offset:offset+width]=0
            offset+=width
        expected=mdp.features(x,207.318,1.635)[:,mdp.OBS_DIM:]
        actual=mdp.features(x[:,:mdp.STATE_DIM+mdp.SENSOR_DIM+2],207.318,1.635,diagnostics_only=True)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_fast_reward_skips_terrain_and_invalidates_on_reset():
    n=3
    native=torch.zeros(n,mdp.SENSOR_DIM);native[:,3]=1;native[:,13]=1
    robot=SimpleNamespace(joint_pos=torch.zeros(n,23),default_joint_pos=torch.zeros(n,23),joint_vel=torch.zeros(n,23))
    env=SimpleNamespace(common_step_counter=0,_stand_reset_generation=0,
        scene={'robot':SimpleNamespace(data=robot)},sim=SimpleNamespace(data=SimpleNamespace(sensordata=native)),
        action_manager=SimpleNamespace(action=torch.zeros(n,23),prev_action=torch.zeros(n,23)),
        cfg=SimpleNamespace(sim=SimpleNamespace(mujoco=SimpleNamespace(gravity=(0,0,-1.635)))),
        _balance_sensor_indices=torch.arange(mdp.SENSOR_DIM),_balance_sensor_slice=slice(0,mdp.SENSOR_DIM),_balance_weight=207.318)
    ground=SimpleNamespace(height=lambda xy:torch.zeros(n,3),scan=lambda pose,center_height=None:torch.zeros(n,25))
    with patch.dict(os.environ,{'WANGYU_REWARD_FAST':'1'}),patch.object(mdp,'position_error',return_value=torch.zeros(n,3)),patch.object(mdp,'surface',return_value=ground) as terrain:
        initial=mdp.signal(env,'joint_pose_squared').clone()
        terrain.assert_not_called()
        robot.joint_pos[1]=.5
        torch.testing.assert_close(mdp.signal(env,'joint_pose_squared'),initial)
        env._stand_reset_generation+=1
        assert mdp.signal(env,'joint_pose_squared')[1]==.25
        terrain.assert_not_called()
        observation=mdp.observation(env)
        assert observation.shape==(n,145)
        terrain.assert_called_once()
        assert observation[1,0]==.5
        env.common_step_counter+=1;robot.joint_pos[1]=.2
        torch.testing.assert_close(mdp.signal(env,'joint_pose_squared')[1],torch.tensor(.04))
        with patch.dict(os.environ,{'WANGYU_REWARD_FAST':'0'}):
            torch.testing.assert_close(mdp.signal(env,'joint_pose_squared')[1],torch.tensor(.04))
