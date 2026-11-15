from pathlib import Path
import mujoco,torch
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg,cartpole_ppo_runner_cfg
from mjlab.entity import EntityCfg,EntityArticulationInfoCfg
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
ROOT=Path(__file__).resolve().parents[1]
def spec():return mujoco.MjSpec.from_file(str(ROOT/'assets/wangyu_proxy.xml'))
def obs(env):
 d=env.scene['robot'].data
 return torch.cat([d.joint_pos,d.joint_vel*.1,d.root_link_lin_vel_b,d.root_link_ang_vel_b,d.projected_gravity_b],-1)
def reward(env):
 d=env.scene['robot'].data
 upright=(-d.projected_gravity_b[:,2]).clamp(0,1)
 return upright*(d.root_link_lin_vel_b[:,2].clamp(0,2)+.1)-.0001*env.action_manager.action.square().sum(-1)
def cfg():
 c=cartpole_balance_env_cfg();c.scene.entities={'robot':EntityCfg(spec_fn=spec,articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=('.*',)),)),init_state=EntityCfg.InitialStateCfg(pos=(0,0,.88),joint_pos={'.*':0},joint_vel={'.*':0}))};c.scene.num_envs=16;c.events={};c.actions={'effort':JointEffortActionCfg(entity_name='robot',actuator_names=('.*',),scale=30.)};c.observations={k:ObservationGroupCfg({'state':ObservationTermCfg(func=obs)}) for k in ('actor','critic')};c.rewards={'jump':RewardTermCfg(func=reward,weight=1)};c.sim.mujoco.disableflags=();c.sim.mujoco.timestep=1/120;c.sim.mujoco.gravity=(0,0,-1.625);c.decimation=2;c.episode_length_s=8;c.viewer.entity_name='robot';c.viewer.body_name='Pelvis';return c
rl=cartpole_ppo_runner_cfg();rl.experiment_name='wangyu_moon_jump';rl.logger='tensorboard';rl.max_iterations=200
register_mjlab_task('Wangyu-Moon-Jump-v0',cfg(),cfg(),rl)

# Independent native-geometry asset; old 20-axis checkpoints remain isolated.
def native_spec():
 return mujoco.MjSpec.from_file(str(ROOT/'assets/wangyu_physics/wangyu.xml'))

def native_skin_spec():
 s=mujoco.MjSpec.from_file(str(ROOT/'assets/wangyu_skinned_robot.xml'))
 for geom in s.geoms: geom.group=3
 # MuJoCo 3.11 attach prefixes skin body references and material names,
 # but leaves skin.material unchanged. This entity is attached as robot/.
 for skin in s.skins: skin.material='robot/'+skin.material
 return s

def native_cfg(play=False):
 import json
 c=cfg()
 manifest=json.loads((ROOT/'assets/wangyu_physics/manifest.json').read_text())
 c.sim.mujoco.integrator='implicitfast'
 c.sim.mujoco.iterations=100
 robot=c.scene.entities['robot']
 robot.spec_fn=native_skin_spec if play else native_spec
 robot.init_state.pos=(0,0,manifest['initial_pelvis_height_m'])
 return c

from copy import deepcopy
native_rl=deepcopy(rl)
native_rl.experiment_name='wangyu_moon_jump_physics_v1'
register_mjlab_task('Wangyu-Moon-Jump-Physics-v1',native_cfg(),native_cfg(play=True),native_rl)
