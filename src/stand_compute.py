"""Fixed-shape stand observation/reward bundle; R3-style owned graph outputs."""
import os
import torch
from tensor_graph import FixedTensorGraph


def tensors(x):
    q, qd = x[:, :20], x[:, 20:40]
    p, quat, vel, action = x[:, 40:43], x[:, 43:47], x[:, 47:53], x[:, 53:73]
    w, v = quat[:, :1], quat[:, 1:]
    def inverse_rotate(vector):
        cross = torch.cross(v, vector, dim=-1)
        return vector - 2*w*cross + 2*torch.cross(v, cross, dim=-1)
    world_gravity = torch.zeros_like(p)
    world_gravity[:, 2] = -1
    gravity = inverse_rotate(world_gravity)
    linear = inverse_rotate(vel[:, :3])
    angular = inverse_rotate(vel[:, 3:])
    obs = torch.cat((q, .1*qd, gravity, linear, angular, p, action), dim=-1)
    error = (gravity[:, :2].square().sum(-1)/.20**2
             + p[:, :2].square().sum(-1)/.15**2 + p[:, 2].square()/.10**2
             + linear.square().sum(-1)/.15**2 + angular.square().sum(-1)/.30**2
             + qd.square().mean(-1)/.75**2 + q.square().mean(-1)/.25**2)
    tilt_rad = torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2])
    tilt_deg = tilt_rad * (180. / 3.141592653589793)
    return torch.cat((obs, torch.exp(-error).unsqueeze(-1), tilt_deg.unsqueeze(-1),
                      tilt_rad.square().unsqueeze(-1)), dim=-1)


def invalidate(env, env_ids):
    """Partial/full reset must invalidate same-control-step cached state."""
    del env_ids
    env._stand_reset_generation = getattr(env, '_stand_reset_generation', 0)+1
    env._stand_bundle_key = None


def position_error(env):
    d = env.scene['robot'].data
    if hasattr(env, '_stand_target_root'):
        p = d.root_link_pos_w-env._stand_target_root
        q = env._stand_target_heading
        c = q[:, 0].square()-q[:, 3].square()
        s = 2*q[:, 0]*q[:, 3]
        return torch.stack((c*p[:, 0]+s*p[:, 1], -s*p[:, 0]+c*p[:, 1], p[:, 2]), -1)
    return d.root_link_pos_w-env.scene.env_origins-d.default_root_state[:, :3]


def bundle(env):
    key = (env.common_step_counter, getattr(env, '_stand_reset_generation', 0))
    if getattr(env, '_stand_bundle_key', None) != key:
        d = env.scene['robot'].data
        pose = d.root_link_pose_w
        p = position_error(env)
        x = torch.cat((d.joint_pos-d.default_joint_pos, d.joint_vel, p,
                       pose[:, 3:7], d.root_link_vel_w, env.action_manager.action), dim=-1)
        use_graph = x.is_cuda and os.environ.get('WANGYU_TENSOR_GRAPH', '1') != '0'
        if use_graph:
            if not hasattr(env, '_stand_tensor_graph'):
                env._stand_tensor_graph = FixedTensorGraph(tensors, x)
            value = env._stand_tensor_graph(x)
        else:
            value = tensors(x)
        env._stand_bundle_key, env._stand_bundle = key, value
    return env._stand_bundle
