"""v2: unchanged 146-input lunar standing task plus height-calibrated upward COM momentum reward."""
from copy import deepcopy
import json
import os
from pathlib import Path
import lunar_finetune_task as base
import jump_mdp as mdp
from jump_prior import JumpRunner as PriorRunner
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.registry import register_mjlab_task

TASK_ID='Wangyu-Moon-Hop-Lunar-v2'
from jump_settings import DEFAULT_JUMP_HEIGHT,REWARD_REVISION,MOMENTUM_EXPONENT,HEIGHT_EXPONENT,validate_height,experiment_base
JUMP_HEIGHT=validate_height(os.environ.get('WANGYU_JUMP_HEIGHT',DEFAULT_JUMP_HEIGHT))
EXPERIMENT_BASE=experiment_base(base.PHYSICS_REVISION,JUMP_HEIGHT)
EXPERIMENT=os.environ.get('WANGYU_JUMP_EXPERIMENT',EXPERIMENT_BASE+'_no_push')
OBS_DIM=base.OBS_DIM
PHYSICS_REVISION=base.PHYSICS_REVISION
OUTPUT_SUBDIR='hop_lunar_v2'
REWARD_WEIGHTS={**base.REWARD_WEIGHTS,'upward_momentum':150.0,'jump_height':100.0}


@requires_model_fields('geom_dataid','geom_pos')
def reset_jump(env,env_ids,distinct_patches=False):
    base.reset_local(env,env_ids,distinct_patches=distinct_patches)
    if hasattr(env,'_jump_height_ready'):
        from mjlab.envs.mdp.events import resolve_env_ids
        ids=resolve_env_ids(env,env_ids)
        env._jump_height_ready[ids]=False
        env._jump_height_baseline[ids]=0.
    env._jump_height_key=None


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    # Preserve every original reward, observation, reset, disturbance, actuator,
    # termination and physics setting. Only the new height baseline resets here.
    cfg.rewards['upward_momentum']=RewardTermCfg(func=mdp.momentum_reward,
        params={'jump_height':JUMP_HEIGHT,'exponent':MOMENTUM_EXPONENT},weight=REWARD_WEIGHTS['upward_momentum'])
    cfg.rewards['jump_height']=RewardTermCfg(func=mdp.height_reward,
        params={'jump_height':JUMP_HEIGHT,'exponent':HEIGHT_EXPONENT},weight=REWARD_WEIGHTS['jump_height'])
    cfg.events['reset_pose'].func=reset_jump
    cfg.metrics['jump_com_rise_m']=MetricsTermCfg(func=mdp.measured_com_rise,reduce='mean')
    cfg.metrics['jump_com_rise_peak_m']=MetricsTermCfg(func=mdp.measured_com_rise,reduce='max')
    cfg.metrics['jump_airborne_com_rise_peak_m']=MetricsTermCfg(func=mdp.measured_airborne_com_rise,reduce='max')
    cfg.metrics['jump_height_airborne_fraction']=MetricsTermCfg(func=mdp.height_airborne_fraction,reduce='mean')
    cfg.metrics['com_upward_speed_m_s']=MetricsTermCfg(func=mdp.upward_speed,reduce='mean')
    cfg.metrics['com_upward_momentum_kg_m_s']=MetricsTermCfg(func=mdp.upward_momentum,reduce='mean')
    cfg.metrics['target_upward_momentum_kg_m_s']=MetricsTermCfg(func=mdp.target_momentum,params={'jump_height':JUMP_HEIGHT},reduce='mean')
    return cfg


class JumpRunner(PriorRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.logger.log_dir:
            from terrain_provenance import save
            destination=Path(self.logger.log_dir);save(destination/'terrain')
            (destination/'hop_config.json').write_text(json.dumps({
                'task':TASK_ID,'obs_dim':OBS_DIM,'reward_revision':REWARD_REVISION,
                'pushes':'pelvis_push' in self.env.unwrapped.cfg.events,
                'rewards':{name:term.weight for name,term in self.env.unwrapped.cfg.rewards.items()},
                'jump_height_m':JUMP_HEIGHT,
                'momentum_exponent':self.env.unwrapped.cfg.rewards['upward_momentum'].params['exponent'],
                'height_exponent':self.env.unwrapped.cfg.rewards['jump_height'].params['exponent'],
                'height_reward':'airborne * expm1(kh * clip(native COM rise/h,0,1))/expm1(kh); baseline=first native supported COM snapshot each episode',
                'target_upward_speed_m_s':mdp.target_upward_speed(self.env.unwrapped,JUMP_HEIGHT),
                'target_momentum_kg_m_s_first_world':float(mdp.target_momentum(self.env.unwrapped,JUMP_HEIGHT)[0]),
                'added_reward':'every control step: expm1(k*x)/expm1(k), x=clamp(current p_up/(native_subtree_mass*sqrt(2*|g|*h)),0,1); no gating; standard dt scaling; RL returns still accumulate',
                'height_semantics':'measured native COM rise opposite gravity from first supported snapshot; not sole clearance; momentum target still uses ideal ballistic rise',
                'sampling_boundary':'native pre-reset COM/contact snapshot per control step; first-support height baseline only, no momentum gate or integration',
                'scope':'Baseline lunar standing configuration plus momentum and actual COM rise exponential rewards'},indent=2)+'\n')

    def load(self,path,*args,**kwargs):
        from terrain_provenance import verify
        verify(path,allow_transfer=os.environ.get('WANGYU_TERRAIN_TRANSFER')=='1')
        return super().load(path,*args,**kwargs)


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=JumpRunner)
