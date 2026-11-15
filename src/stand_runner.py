"""Task-local GPU episode aggregation; preserves upstream 100-episode window."""
import os
import json
from pathlib import Path
import torch
from rsl_rl.utils.logger import Logger
from mjlab.rl import MjlabOnPolicyRunner


class StandLogger(Logger):
    @classmethod
    def from_logger(cls, old):
        obj = cls.__new__(cls)
        obj.__dict__.update(old.__dict__)
        shape = (obj.cfg['num_steps_per_env'], obj.num_envs)
        obj._done = torch.empty(shape, dtype=torch.bool, device=obj.device)
        obj._rewards = torch.empty(shape, device=obj.device)
        obj._lengths = torch.empty(shape, device=obj.device)
        obj._step_rewards = torch.empty(shape, device=obj.device)
        obj._step_dt = 1.
        obj._cursor = 0
        return obj

    def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None):
        if self.writer is None:
            return
        if intrinsic_rewards is not None:
            raise ValueError('StandLogger is for this non-RND standing task only')
        if self._cursor == self._done.shape[0]:
            raise RuntimeError('Episode log buffer must be flushed after each rollout')
        info = extras.get('episode', extras.get('log'))
        if info is not None:
            self.ep_extras.append({k: v.clone() if isinstance(v, torch.Tensor) else v
                                   for k, v in info.items()})
        self.cur_reward_sum.add_(rewards)
        self.cur_episode_length.add_(1)
        done = dones > 0
        self._step_rewards[self._cursor].copy_(rewards)
        self._done[self._cursor].copy_(done)
        self._rewards[self._cursor].copy_(self.cur_reward_sum)
        self._lengths[self._cursor].copy_(self.cur_episode_length)
        self.cur_reward_sum.masked_fill_(done, 0)
        self.cur_episode_length.masked_fill_(done, 0)
        self._cursor += 1

    def flush_episodes(self):
        # Row-major preserves the upstream step-then-environment completion order.
        done = self._done[:self._cursor].reshape(-1)
        ids = done.nonzero(as_tuple=False).flatten()[-100:]
        pairs = torch.stack((self._rewards[:self._cursor].reshape(-1)[ids],
                             self._lengths[:self._cursor].reshape(-1)[ids]), dim=-1).cpu().tolist()
        self.rewbuffer.extend(row[0] for row in pairs)
        self.lenbuffer.extend(row[1] for row in pairs)
        self._cursor = 0

    def log(self, *args, **kwargs):
        if self.writer is not None and self._cursor:
            samples=self._step_rewards[:self._cursor]
            stats=torch.stack((samples.mean(),(samples>0).float().mean(),
                               self._done[:self._cursor].sum().float())).cpu().tolist()
            it=kwargs['it'] if 'it' in kwargs else args[0]
            self.writer.add_scalar('Rollout/mean_step_reward',stats[0],it)
            self.writer.add_scalar('Rollout/mean_reward_rate',stats[0]/self._step_dt,it)
            self.writer.add_scalar('Rollout/positive_reward_fraction',stats[1],it)
            self.writer.add_scalar('Rollout/completed_episodes',stats[2],it)
        self.flush_episodes()
        return super().log(*args, **kwargs)


class StandRunner(MjlabOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.environ.get('WANGYU_FAST_LOG', '1') != '0':
            self.logger = StandLogger.from_logger(self.logger)
            self.logger._step_dt = float(self.env.unwrapped.step_dt)
        options={name:os.environ.get(name,default) for name,default in
                 (('WANGYU_REWARD_FAST','0'),('WANGYU_TENSOR_GRAPH','1'),('WANGYU_FAST_LOG','1'))}
        print('[Training compute] '+', '.join(f'{k}={v}' for k,v in options.items()),flush=True)
        if self.logger.log_dir:
            directory=Path(self.logger.log_dir);directory.mkdir(parents=True,exist_ok=True)
            (directory/'compute_options.json').write_text(json.dumps(options,indent=2)+'\n')
