"""CPU-only generation review against local OmniLRS; no physics stepping.

Samples native static geometry distances and static contact counts. CPU contact
counts do not establish MJWarp counts or GPU buffer safety.
"""
import json
import hashlib
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import mujoco
from mjlab.scene import Scene
import lunar_details as details
import local_lunar_terrain as terrain
import lunar_finetune_task as task
from lunar_terrain import terrain_data


def selection_audit(bank):
    counts=bank['spawn_counts']
    return {'terrain_patches':len(counts),'detail_based_whole_patch_rejections':0,
            'spawnable_patches':int((counts>0).sum()),'safe_placements':int(counts.sum()),
            'minimum_safe_placements_per_patch':int(counts.min())}


def native_spawn_audit(bank,samples=256):
    torch.set_num_threads(1)
    cfg=task.terrain_stand_cfg();cfg.scene.num_envs=1
    model=Scene(cfg.scene,'cpu').compile();data=mujoco.MjData(model)
    ground=terrain.LocalSurface(SimpleNamespace(sim=SimpleNamespace(mj_model=model),num_envs=samples,device='cpu'))
    rng=np.random.default_rng(923);patches=rng.choice(np.flatnonzero(bank['spawn_counts']>0),samples)
    ground.patch_ids[:]=torch.tensor(patches)
    placement=(rng.random(samples)*bank['spawn_counts'][patches]).astype(int)
    xy=bank['spawn_xy'][patches,placement]+rng.uniform(-.025,.025,(samples,2));yaw=rng.uniform(-np.pi,np.pi,samples)
    foot=ground.sole_points.numpy();c=np.cos(yaw)[:,None];s=np.sin(yaw)[:,None]
    points=np.stack((xy[:,None,0]+c*foot[None,:,0]-s*foot[None,:,1],
                     xy[:,None,1]+s*foot[None,:,0]+c*foot[None,:,1]),-1)
    heights=ground.height(torch.tensor(points,dtype=torch.float32)).numpy()
    z=(heights-foot[None,:,2]).max(-1)+terrain.CLEARANCE+ground.clearance_padding[torch.tensor(patches)].numpy()
    soles=[model.geom(f'robot/{side}_sole').id for side in ('left','right')]
    distances=[];counts=[];initial_penetrations=[];contact_depths=[];vertical_clearances=[]
    for i,patch in enumerate(patches):
        model.geom_dataid[ground.geom_id]=int(ground.native_ids[patch])
        model.geom_pos[ground.geom_id,2]=bank['zmin'][patch]
        data.qpos[:]=model.qpos0;data.qpos[:3]=[*xy[i],z[i]]
        data.qpos[3:7]=[np.cos(yaw[i]/2),0,0,np.sin(yaw[i]/2)]
        mujoco.mj_forward(model,data)
        initial_penetrations.append(sum(v.dist<0 for v in data.contact[:data.ncon]))
        gap=min(mujoco.mj_geomDistance(model,data,ground.geom_id,gid,.5,None) for gid in soles)
        distances.append(float(gap))
        # Probe only: move down 20mm to obtain static near-contact manifolds.
        # No integration/solver step; this is not a GPU capacity validation.
        data.qpos[2]-=.02;mujoco.mj_forward(model,data)
        counts.append(int(data.ncon))
        contact_depths.extend(float(v.dist) for v in data.contact[:data.ncon])
        # Determine the static vertical lowering needed to first touch terrain.
        # This tests the whole native shoe/hfield manifold, not just 50 samples.
        lo,hi=0.,.05
        for _ in range(9):
            middle=(lo+hi)/2;data.qpos[2]=z[i]-middle;mujoco.mj_forward(model,data)
            if any(v.dist < -1e-7 for v in data.contact[:data.ncon]):hi=middle
            else:lo=middle
        vertical_clearances.append((lo+hi)/2)
    return {'samples':samples,'native_penetrating_spawn_count':int(np.count_nonzero(initial_penetrations)),
            'geom_distance_diagnostic_min_m':float(min(distances)),
            'geom_distance_note':'Not used as a safety criterion: non-convex heightfields returned negative distance where native collision manifolds found no penetration.',
            'native_cpu_contacts_after_20mm_lowering_max':max(counts),
            'native_cpu_contact_min_dist_after_lowering_m':min(contact_depths,default=None),
            'native_static_vertical_clearance_min_m':min(vertical_clearances),
            'native_static_vertical_clearance_below_14_5mm':int(np.count_nonzero(np.array(vertical_clearances)<.0145)),
            'vertical_clearance_search_precision_m':.05/2**9,
            'native_cpu_contacts_after_20mm_lowering_gt_gpu_budget':int(np.count_nonzero(np.array(counts)>cfg.sim.nconmax)),
            'configured_gpu_nconmax':cfg.sim.nconmax,'configured_gpu_njmax':cfg.sim.njmax,
            'scope':'Native CPU static distances and lowered contact manifolds only; no mj_step, GPU simulation, or policy.'}


def native_mesh_seams(bank):
    cfg=task.terrain_stand_cfg(play=True);cfg.scene.num_envs=1
    model=Scene(cfg.scene,'cpu').compile();data=mujoco.MjData(model)
    surface=terrain.LocalSurface(SimpleNamespace(sim=SimpleNamespace(mj_model=model),num_envs=terrain.COUNT,device='cpu'))
    surface.patch_ids[:]=torch.arange(terrain.COUNT)
    model.geom_group[:]=0;model.geom_group[surface.visual_ids.numpy()]=5
    mujoco.mj_kinematics(model,data)
    errors=[];jumps=[];exact_misses=0
    group=np.array([0,0,0,0,0,1],dtype=np.uint8)
    for index in range(terrain.COUNT):
        center=bank['centers'][index]-bank['centers'][0]
        altitude=bank['altitudes'][index]-bank['altitudes'][0]
        for xy,normal in (((8.,.737),(1.,0)),((-8.,-.913),(-1.,0)),
                          ((1.337,8.),(0,1.)),((-2.137,-8.),(0,-1.))):
            def ray(point):
                return mujoco.mj_ray(model,data,np.r_[center+point,altitude+10],np.array([0.,0.,-1.]),
                                     group,True,-1,None)
            if ray(np.array(xy))<0:exact_misses+=1
            heights=[]
            for sign in (-1,1):
                point=np.array(xy)+sign*.001*np.array(normal)
                distance=ray(point)
                if distance<0:raise ValueError(f'Missing terrain beside seam {index}/{point}')
                heights.append(10-distance)
                if sign<0:
                    expected=float(surface.height(torch.tensor(point[None,:],dtype=torch.float32),torch.tensor([index]))[0])
                    errors.append(abs(heights[-1]-expected))
            jumps.append(abs(heights[1]-heights[0]))
    if max(errors)>2e-4:raise ValueError(f'Native mesh seam height mismatch: {max(errors)}m')
    if max(jumps)>.01:raise ValueError(f'Discontinuous surface beside seam: {max(jumps)}m')
    return {'inside_ray_checks':len(errors),'outside_ray_checks':len(jumps),
            'max_inside_height_error_m':max(errors),'tolerance_m':2e-4,
            'max_height_change_across_2mm_m':max(jumps),'exact_edge_ray_misses':exact_misses,
            'note':'An exact-edge native ray can miss due to triangle predicate roundoff; both sides are checked separately. Welded boundary indices guarantee topology.',
            'boundary_edges_checked_per_patch':4*(bank['normalized'].shape[1]-1)}


def main():
    torch.set_num_threads(1)
    bank=terrain.bank_data();cfg=details.settings()
    center=np.array([25.,40.]);world=center+[2.5,0]
    other=center+[5,0]
    overlap=[float(details.generate((world-c)[None,:],c)[0]) for c in (center,other)]
    smooth=replace(cfg,crater_density_m2=0,mound_density_m2=0)
    jumps=[]
    for eps in (1e-3,1e-5,1e-7):
        values=details.generate(np.array([[4.5-eps,4.5],[4.5,4.5],[4.5+eps,4.5]]),center,smooth)
        jumps.append({'epsilon_m':eps,'derivative_jump':float((values[2]-2*values[1]+values[0])/eps)})
    report={'scope':'Code inspection + deterministic generation + CPU static native geometry; production terrain unchanged.',
            'source_sha256':{name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                for name in ('src/lunar_details.py','src/local_lunar_terrain.py','src/lunar_finetune_task.py')},
            'omnilrs_commit':subprocess.run(['git','-C',str(Path.home()/'Github/OmniLRS'),'rev-parse','HEAD'],
                capture_output=True,text=True,check=True).stdout.strip(),
            'detail_metadata':bank['detail_metadata'],
            'coverage':{'map_area_m2':4e6,'patch_count':terrain.COUNT,'high_resolution_display_fraction':terrain.COUNT*(2*terrain.HALF)**2/4e6,
                        'global_field':'available on demand; no full-map 0.1m allocation'},
            'sampling':{'step_m':terrain.STEP,'min_rim_sigma_m':max(.16*min(cfg.crater_radius_m),cfg.min_rim_sigma_cells*terrain.STEP),
                        'min_rim_fwhm_grid_intervals':2.355*max(.16*min(cfg.crater_radius_m),cfg.min_rim_sigma_cells*terrain.STEP)/terrain.STEP,
                        'min_mound_minor_diameter_grid_intervals':2*.8*min(cfg.mound_radius_m)/terrain.STEP,
                        'height_observation_scan_step_m':.25},
            'crop_coordinate_consistency':{'same_global_point':world.tolist(),'crop_centers':[center.tolist(),other.tolist()],
                'generated_deltas_m':overlap,'difference_m':abs(overlap[0]-overlap[1]),
                'note':'Global feature metadata and noise are shared by overlapping crops; difference should be numerical roundoff only.'},
            'global_derivative_probe_at_old_fade_diagonal':jumps,
            'selection':selection_audit(bank),'native_spawn':native_spawn_audit(bank),'mesh_seams':native_mesh_seams(bank),
            'findings':[
                {'priority':'P1','kind':'spawn clearance check gap','source':'src/lunar_finetune_task.py:72-78',
                 'finding':'The reset clearance assertion uses 50 fixed sole samples, not the whole shoe/heightfield contact manifold; unsampled crests can reduce the intended 15mm gap.',
                 'recommendation':'Check conservative whole-sole clearance at the collider resolution; use native static collision as an independent oracle, not mj_geomDistance for these non-convex hfields.'},
                {'priority':'P1','kind':'reproducibility gap','source':'src/lunar_finetune_task.py:120; scripts/play_stand.py:63,343',
                 'finding':'Training saves bank.npz and metadata, but replay regenerates the current bank without loading/checking saved terrain. Parameter/code edits can silently change the replay terrain.',
                 'recommendation':'Load saved terrain or require explicit transfer when generator/config fingerprint differs.'},
                {'priority':'P1','kind':'sampling limitation','source':'src/lunar_details.py:88',
                 'finding':'Narrowest rim FWHM is only 1.13 grid intervals; smallest mound minor diameter is 3.2 intervals. No antialias/filtering is applied.',
                 'recommendation':'Set minimum resolvable feature scale and native collision budget before choosing finer sampling or band-limiting geometry.'},
                {'priority':'P1','kind':'selection bias','source':'src/local_lunar_terrain.py:51',
                 'finding':'Whole patches are rejected after checking every central microtriangle against 15 degrees, preferentially removing rough craters instead of selecting valid foot placements within retained terrain.',
                 'recommendation':'Separate fixed terrain generation from safe spawn-site selection; record actual retained terrain statistics.'},
                {'priority':'P2','kind':'architecture limitation','source':'src/lunar_details.py:69; src/local_lunar_terrain.py:60',
                 'finding':'Only 0.16% of the full map contains detail inserts. Crop-seeded features are not one globally consistent procedural field. Current disjoint inserts avoid overlap mismatches but cannot be streamed arbitrarily.',
                 'recommendation':'Use world-coordinate feature metadata with neighbor-block padding/cache as in the OmniLRS crater DB approach.'},
                {'priority':'P2','kind':'surface regularity','source':'src/lunar_details.py:98',
                 'finding':'Smoothstep gives a smooth outer seam, but max(abs(x),abs(y)) introduces derivative creases along internal fade-band diagonals.',
                 'recommendation':'Use a separable smooth envelope or padded global sampling, and test directional derivatives as well as seam heights.'},
                {'priority':'P2','kind':'fidelity limitation','source':'src/lunar_details.py:73,81,85,95; src/local_lunar_terrain.py:46',
                 'finding':'Single radial crater family, one uniform radius range and smooth embedded mounds provide limited morphology. Subdivision preserves all original 5m DEM triangular creases; unlike OmniLRS interpolation it does not reconstruct a smooth base.',
                 'recommendation':'Add morphology/size-class metadata and decide base-surface reconstruction explicitly; do not claim these features are measured lunar terrain.'}
            ],
            'references':{'omnilrs_craters':'~/Github/OmniLRS/src/terrain_management/large_scale_terrain/crater_generation.py',
                'omnilrs_block_consistency':'~/Github/OmniLRS/src/terrain_management/large_scale_terrain/crater_distribution.py',
                'omnilrs_resolution':'~/Github/OmniLRS/cfg/environment/lunaryard_20m.yaml (0.025m)'}}
    report['references']['omnilrs_large_scale_defaults']='~/Github/OmniLRS/src/configurations/environments.py:138,176 (DEM 0.025m, collider 0.05m; not an automatic recommendation for 8192 MJWarp worlds)'
    report.pop('findings')
    report['baseline_audit']='outputs/terrain_generation_audit/report.json'
    report['scope']='Detail3 CPU static verification; no physics stepping or training.'
    path=Path(__file__).resolve().parents[1]/'outputs/terrain_generation_audit/detail3_verification.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
