"""Bounded standing contracts, R3 graph equivalence, partial reset and log order."""
from pathlib import Path
import sys,json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import stand_task as task
from stand_compute import bundle,tensors
from tensor_graph import FixedTensorGraph
from stand_runner import StandLogger
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.utils.logger import Logger

torch.set_num_threads(1)
device='cuda:0'
# Pure tensor graph: input changes, no output aliasing, and eager equivalence.
x=torch.randn((4,73),device=device)*.1
x[:,43:47]=torch.nn.functional.normalize(torch.randn((4,4),device=device),dim=-1)
g=FixedTensorGraph(tensors,x)
first=g(x);saved=first.clone()
torch.testing.assert_close(first,tensors(x))
torch.testing.assert_close(g(x+.001),tensors(x+.001))
torch.testing.assert_close(first,saved)
# Preserve upstream step-then-world completion order and its last-100 window.
cfg={'num_steps_per_env':8,'algorithm':{}}
old=Logger(None,cfg,None,32,False,1,0,device);new=StandLogger.from_logger(old)
old.writer=object();new.writer=object()
for step in range(8):
 rewards=torch.arange(32,device=device,dtype=torch.float32)*.01+step
 done=(torch.arange(32,device=device)+step)%3==0
 old.process_env_step(rewards,done,{})
 new.process_env_step(rewards,done,{})
new.flush_episodes()
assert list(old.rewbuffer)==list(new.rewbuffer) and list(old.lenbuffer)==list(new.lenbuffer)
torch.testing.assert_close(old.cur_reward_sum,new.cur_reward_sum)
# Real physics/action/reset: only foot-ground contact candidates, no root forces.
c=task.stand_cfg();assert c.scene.num_envs==8192
c.scene.num_envs=4
e=ManagerBasedRlEnv(cfg=c,device=device)
try:
 obs,_=e.reset();d=e.scene['robot'].data;m=e.sim.mj_model
 assert obs['actor'].shape==(4,72) and e.action_manager.total_action_dim==20
 np.testing.assert_allclose(m.opt.gravity,[0,0,-9.81/6],atol=1e-12)
 torch.testing.assert_close(task.pose_error(e),torch.zeros((4,3),device=device),atol=1e-6,rtol=0)
 torch.testing.assert_close(task.stability(e),torch.ones(4,device=device),atol=1e-5,rtol=0)
 expected=torch.cat((d.joint_pos-d.default_joint_pos,.1*d.joint_vel,d.projected_gravity_b,
                     d.root_link_lin_vel_b,d.root_link_ang_vel_b,task.pose_error(e),e.action_manager.action),-1)
 torch.testing.assert_close(task.observation(e),expected,atol=1e-6,rtol=1e-6)
 geom_names=[m.geom(i).name for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]]
 assert all(n.endswith(('left_sole','right_sole','terrain')) or 'terrain' in n for n in geom_names),geom_names
 assert m.nexclude==1
 assert not m.neq and m.nskin==0
 e.action_manager.process_action(torch.ones((4,20),device=device)*10)
 term=e.action_manager.get_term('position')
 assert term._processed_actions.abs().max()<=task.ACTION_LIMIT+1e-6
 for _ in range(8):obs,*_=e.step(torch.zeros((4,20),device=device))
 before=d.root_link_pos_w.clone()
 e.reset(env_ids=torch.tensor([0],device=device))
 torch.testing.assert_close(d.root_link_pos_w[1:],before[1:])
 torch.testing.assert_close(task.pose_error(e)[0],torch.zeros(3,device=device),atol=1e-6,rtol=0)
 torch.testing.assert_close(task.observation(e)[0,:20],torch.zeros(20,device=device))
 assert torch.isfinite(obs['actor']).all()
 report={'default_envs':8192,'observations':72,'actions':20,'gravity_m_s2':-task.GRAVITY,'physics_cuda_graph':e.sim.use_cuda_graph,
         'tensor_graph_eager_parity':True,'partial_reset_cache_invalidated':True,'log_order_and_window_parity':True,
         'processed_target_clip_rad':task.ACTION_LIMIT,'collision_geoms':geom_names,'no_foot_self_collision':True}
 path=Path('outputs/stand/contracts.json');path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
finally:e.close()
