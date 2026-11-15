"""CPU preparation of distinct lunar instances and safe randomized root poses."""
import json
from pathlib import Path
import numpy as np
from sim2sim_contract import height


def sample_root(grid,pool,points,padding,rng,yaw_offset=0.,limit=2.,excluded=None,separation=1.5):
    """Use the production spawn pool + yaw + native whole-sole cover bound."""
    for _ in range(512):
        xy=pool[int(rng.integers(len(pool)))]+rng.uniform(-.025,.025,2)
        if excluded is not None and len(excluded) and (np.linalg.norm(np.asarray(excluded)[:,:2]-xy,axis=-1)<separation).any():continue
        yaw=rng.uniform(-np.pi,np.pi)+yaw_offset;c,s=np.cos(yaw),np.sin(yaw)
        world=points[:,:2]@np.array([[c,s],[-s,c]])+xy
        if np.any(np.abs(world)>limit):continue
        terrain=height(world,grid,8.)
        z=float(np.max(terrain-points[:,2])+.015+padding)
        gaps=z+points[:,2]-terrain
        if np.isfinite(gaps).all() and gaps.min()>=.0145 and gaps.max()<=.20:
            return np.array([*xy,z,np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
    raise ValueError('No legal randomized/separated root pose after 512 bounded attempts')


def sample_layout(grid,pool,points,padding,count,rng,yaw_offset=0.):
    # Random greedy placement can use up a small safe pocket. Retry the layout
    # before giving up; never relax spacing/clearance or leave the moon patch.
    for _ in range(16):
        roots=[]
        try:
            for _ in range(count):
                roots.append(sample_root(grid,pool,points,padding,rng,yaw_offset,limit=7.,excluded=roots))
            return np.array(roots)
        except ValueError:continue
    raise ValueError(f'Cannot place {count} separated humans on this safe moon patch; reduce --num-envs or select another --patch')


def prepare_instances(args,manifest,bank,points,cover):
    from scipy.ndimage import maximum_filter
    count=args.num_envs;patch=args.patch
    if manifest['terrain']['mode']!='lunar':
        raise ValueError('Multi-human shared scene currently requires --terrain moon')
    seed=int(args.seed) if args.seed is not None else int(np.random.SeedSequence().entropy)
    rng=np.random.default_rng(seed);chosen=np.full(count,patch,dtype=int)
    grid=bank['normalized'][patch]*bank['height_range'][patch]+bank['zmin'][patch]
    grids=np.broadcast_to(grid,(count,*grid.shape));offsets=np.zeros((count,3))
    spacing=16./(grid.shape[0]-1)
    z00,z10,z01,z11=grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1],grid[1:,1:]
    slope=np.maximum(np.hypot(z11-z01,z01-z00),np.hypot(z10-z00,z11-z10))/spacing
    # Same .6m rest-sole envelope as the training pool, expanded over the
    # interior of this ONE patch. Leave >2m to the finite terrain edge.
    maximum=maximum_filter(slope,size=2*int(np.ceil(.6/spacing))+1,mode='nearest')
    axis=np.linspace(-5.5,5.5,23);x,y=np.meshgrid(axis,axis);pool=np.stack((x,y),-1).reshape(-1,2)
    ij=np.floor((pool+8)/spacing).astype(int)
    pool=pool[maximum[ij[:,1],ij[:,0]]<=np.tan(np.deg2rad(15))]
    if not len(pool):raise ValueError('Selected moon patch has no safe shared-area spawn pool')
    counts=np.full(count,len(pool),dtype=int);spawns=np.broadcast_to(pool,(count,*pool.shape))
    dx=np.abs(np.diff(grids,axis=2)).max((1,2))/spacing
    dy=np.abs(np.diff(grids,axis=1)).max((1,2))/spacing
    padding=np.hypot(dx,dy)*cover+2e-6
    roots=sample_layout(grid,pool,points,padding[0],count,rng,np.deg2rad(args.yaw))
    np.savez_compressed(args.bundle/'instances.npz',grids=grids,offsets=offsets,spawn_xy=spawns,
                        spawn_counts=counts,foot_points=points,padding=padding,roots=roots,patches=chosen)
    manifest['num_envs']=count
    manifest['version']=2
    manifest['instances']={'seed':seed,'patches':chosen.tolist(),'offsets':offsets.tolist(),
                           'local_root_spawns':roots.tolist(),'yaw_offset_deg':args.yaw,
                           'layout_revision':'shared_patch_v1','separation_m':1.5,'shared_patch':int(patch),'patch_size_m':16.,
                           'layout':'ONE 16m moon patch and ONE ground collider; random separated safe XY/yaw per reset'}


def instance_manifest(manifest,index,grids,roots):
    """Keep policy observations in training-local coordinates for each actor."""
    value=dict(manifest);value['root_spawn']=roots[index].tolist()
    value['terrain']=dict(manifest['terrain'],patch=manifest['instances']['patches'][index])
    return value
