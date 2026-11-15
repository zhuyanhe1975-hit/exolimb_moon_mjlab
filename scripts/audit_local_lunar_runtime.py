"""Bounded native GPU ground-pose/contact regression; no learning or viewer.

Run explicitly with the shared Python. All 64 DEM variants are exercised twice
through real reset + native forward, with soles lowered 30mm to probe contacts.
This is a collision plumbing test, not a standing-policy evaluation.
"""
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends
import lunar_finetune_task as task
from local_lunar_terrain import COUNT


def main():
    configure_torch_backends()
    cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=COUNT
    cfg.events.pop('pelvis_push',None)
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
    results=[]
    try:
        env.reset()
        ground=env._local_lunar_surface;gid=ground.geom_id
        for offset in (0,17):
            chosen=(torch.arange(COUNT,device=env.device)+offset)%COUNT
            with patch.object(task.torch,'randint',return_value=chosen):
                task.reset_local(env,None)
            env.sim.forward()
            actual=env.sim.data.geom_xpos[:,gid]
            expected=env.sim.model.geom_pos[:,gid]
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
            if ground.visual_ids is not None:
                torch.testing.assert_close(env.sim.data.geom_xpos[:,ground.visual_ids],
                    env.sim.model.geom_pos[:,ground.visual_ids],atol=1e-5,rtol=0)
            env.sim.data.qpos[:,2]-=.03
            env.sim.forward();env.scene.update(dt=0)
            sensor=env.scene['support_contacts'];sensor.update(dt=0)
            found=(sensor.data.found>0).any(-1)
            assert bool(found.all()),f'Missing sole contacts in worlds {torch.where(~found)[0].tolist()}'
            results.append({'offset':offset,'worlds_with_contact':int(found.sum()),
                            'max_ground_pose_error_m':float((actual-expected).abs().max())})
        report={'physics_revision':task.PHYSICS_REVISION,'passed':True,'checks':results,
                'scope':'All 64 native GPU heightfield poses and forced sole contact presence, twice; not policy stability.'}
        path=Path(__file__).resolve().parents[1]/'outputs/stand_lunar_v15/groundsync_audit.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        env.close()


if __name__=='__main__':main()
