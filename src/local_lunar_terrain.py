"""Site20 local, center-relative DEM bank; shared native hfields per world.

The 5m DEM is reconstructed by a continuous cubic spline; a global procedural
field is sampled at .1m. Native fine patches share their boundary with a coarse
full-map mesh; no per-crop fade, cracks or independent crater realization.
All assets share XY bounds. The largest vertical span is the default collider,
providing conservative broadphase bounds. Native dataid and geom Z vary on reset.
"""
from functools import lru_cache
import numpy as np
import mujoco
import torch
from lunar_terrain import terrain_data,Surface
import lunar_details

HALF=8.
STEP=lunar_details.settings().resolution_m
COUNT=64
JITTER=1.25
CLEARANCE=.015
REVISION='native_groundsync1_detail3'


def sole_sampling(model,spacing=.025):
    """Native rest transforms + a covering grid for the whole box sole.

    No native batched whole-hfield distance oracle is suitable here (see audit).
    A Lipschitz bound for native planar hfield triangles bounds between-sample
    height, so finite sampling does not pretend to be an exact surface test.
    """
    data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
    points=[];cover=0.
    for side in ('left','right'):
        gid=model.geom(f'robot/{side}_sole').id
        if not np.allclose(data.geom_xmat[gid].reshape(3,3)[:,2],[0,0,1],atol=1e-8):
            raise ValueError('Certified spawn sampling requires horizontal rest soles')
        size=model.geom_size[gid];nx=int(np.ceil(2*size[0]/spacing));ny=int(np.ceil(2*size[1]/spacing))
        x,y=np.meshgrid(np.linspace(-size[0],size[0],nx+1),np.linspace(-size[1],size[1],ny+1))
        local=np.stack((x,y,np.full_like(x,-size[2])),-1).reshape(-1,3)
        points.append(local@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]-data.qpos[:3])
        cover=max(cover,float(np.hypot(size[0]/nx,size[1]/ny)))
    return np.concatenate(points),cover


def source_height(xy,meta,grid):
    scale=2*meta['half_extent_m']/(grid.shape[0]-1)
    ij=(xy+meta['half_extent_m'])/scale
    col=np.floor(ij[...,0]).astype(int);row=np.floor(ij[...,1]).astype(int)
    u=ij[...,0]-col;v=ij[...,1]-row
    z00=grid[row,col];z10=grid[row,col+1];z01=grid[row+1,col];z11=grid[row+1,col+1]
    a=z00+u*(z11-z01)+v*(z01-z00)
    b=z00+u*(z10-z00)+v*(z11-z10)
    return np.where(v>=u,a,b)*meta['height_range_m']+meta['zmin_m']


@lru_cache(maxsize=1)
def base_interpolator():
    # CPU asset preparation: MuJoCo supplies triangle collision surfaces, not
    # a continuous reconstruction of a low-resolution raster DEM.
    from scipy.interpolate import RectBivariateSpline
    meta,grid,_=terrain_data()
    axis=np.linspace(-meta['half_extent_m'],meta['half_extent_m'],grid.shape[0])
    return RectBivariateSpline(axis,axis,grid*meta['height_range_m']+meta['zmin_m'],s=0)


def base_height(xy):
    xy=np.asarray(xy)
    meta,_,_=terrain_data()
    if not np.isfinite(xy).all() or (abs(xy)>meta['half_extent_m']).any():
        raise ValueError('Base DEM query outside finite source map')
    return base_interpolator().ev(xy[...,1].ravel(),xy[...,0].ravel()).reshape(xy.shape[:-1])


def safe_spawn_points(patch):
    """Select placements within an unchanged terrain, never reject its detail.

    .6m envelope includes the source rest sole radius (verified <=.5m), .025m
    reset jitter, plus grid cell rounding. Native contact safety remains guarded
    by the whole-sole covering bound in reset_local.
    """
    from scipy.ndimage import maximum_filter
    a=patch[:-1,:-1];b=patch[:-1,1:];c=patch[1:,:-1];d=patch[1:,1:]
    slope=np.maximum(np.hypot(d-c,c-a),np.hypot(b-a,d-b))/STEP
    radius=int(np.ceil(.6/STEP))
    maximum=maximum_filter(slope,size=2*radius+1,mode='nearest')
    axis=np.linspace(-1.2,1.2,13);x,y=np.meshgrid(axis,axis)
    xy=np.stack((x,y),-1).reshape(-1,2)
    ij=np.floor((xy+HALF)/STEP).astype(int)
    return xy[maximum[ij[:,1],ij[:,0]]<=np.tan(np.deg2rad(15))]


@lru_cache(maxsize=1)
def bank_data():
    meta,grid,spawn=terrain_data()
    centers=np.unique(np.round(spawn/5.)*5.,axis=0)
    centers=centers[(abs(centers)+HALF<=meta['half_extent_m']-20).all(-1)]
    rng=np.random.default_rng(42);rng.shuffle(centers)
    axis=np.linspace(-HALF,HALF,round(2*HALF/STEP)+1)
    xx,yy=np.meshgrid(axis,axis);local=np.stack((xx,yy),-1)
    selected=[];patches=[];altitudes=[];details=[];spawns=[]
    for center in centers:
        if any(np.max(abs(center-previous))<2*HALF+2 for previous in selected):continue
        altitude=float(base_height(center))
        detail=lunar_details.generate(local,center)
        patch=base_height(local+center)-altitude+detail
        patches.append(patch);selected.append(center);altitudes.append(altitude);details.append(detail)
        spawns.append(safe_spawn_points(patch))
        if len(patches)==COUNT:break
    if len(patches)!=COUNT:raise ValueError('Not enough source DEM patches')
    patches=np.array(patches)
    low=patches.min((1,2));span=patches.max((1,2))-low
    order=np.argsort(-span)
    patches=patches[order];low=low[order];span=span[order]
    normalized=((patches-low[:,None,None])/span[:,None,None]).astype(np.float32)
    counts=np.array([len(spawns[i]) for i in order],dtype=np.int32)
    if not counts.any():raise ValueError('Terrain bank has no safe standing placements')
    spawn_xy=np.zeros((COUNT,169,2),dtype=np.float32)
    for i,index in enumerate(order):spawn_xy[i,:counts[i]]=spawns[index]
    return {'source':meta,'centers':np.array(selected)[order],'altitudes':np.array(altitudes)[order],
            'zmin':low,'height_range':span,'normalized':normalized,
            'detail':np.array(details)[order],'detail_metadata':lunar_details.metadata(),
            'spawn_xy':spawn_xy,'spawn_counts':counts,
            'view_zmin':low,'view_height_range':span,'view_normalized':normalized}



def add_full_view(spec,parent,bank):
    """Full-map coarse mesh with conforming boundaries around fine hfields.

    Background coarse samples retain measured DEM nodes. Short fine boundary
    segments are enforced and checked in the triangulation, while hole interiors
    are filled with the exact 16m collision assets. Far-field is a display LOD.
    """
    from scipy.spatial import Delaunay
    meta,grid,_=terrain_data();n=grid.shape[0]
    axis=np.linspace(-meta['half_extent_m'],meta['half_extent_m'],n)
    x,y=np.meshgrid(axis,axis)
    xy=np.stack((x,y),-1).reshape(-1,2)
    keep=np.ones(len(xy),dtype=bool)
    # Discard coarse vertices on/inside each hole. Native hfield edge samples
    # provide the sole authoritative boundary values and connectivity.
    for center in bank['centers']:keep&=~(abs(xy-center)<=HALF+1e-9).all(-1)
    vertices=[np.column_stack((xy[keep],(grid*meta['height_range_m']+meta['zmin_m']).ravel()[keep]))]
    fine_axis=np.linspace(-HALF,HALF,bank['normalized'].shape[1])
    fx,fy=np.meshgrid(fine_axis,fine_axis);edge=(abs(fx)==HALF)|(abs(fy)==HALF)
    for index,center in enumerate(bank['centers']):
        z=bank['normalized'][index]*bank['height_range'][index]+bank['zmin'][index]+bank['altitudes'][index]
        vertices.append(np.column_stack((fx[edge]+center[0],fy[edge]+center[1],z[edge])))
    vertices=np.concatenate(vertices)
    faces=Delaunay(vertices[:,:2]).simplices
    centroid=vertices[faces,:2].mean(1);keep=np.ones(len(faces),dtype=bool)
    for center in bank['centers']:keep&=~(abs(centroid-center)<HALF-1e-8).all(-1)
    faces=faces[keep]
    # All short boundary edges must survive in the far-field mesh. Their
    # empty diameter disks make them Gabriel/Delaunay edges, verified here.
    all_edges=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    codes,multiplicity=np.unique(all_edges[:,0].astype(np.int64)*len(vertices)+all_edges[:,1],return_counts=True)
    # Boundary point indices are stored in row-major order, not perimeter order.
    start=len(vertices)-COUNT*int(edge.sum())
    perimeter=np.arctan2(fy[edge],fx[edge]).argsort()
    for index in range(COUNT):
        ids=start+index*int(edge.sum())+perimeter
        other=np.roll(ids,-1)
        wanted=np.minimum(ids,other).astype(np.int64)*len(vertices)+np.maximum(ids,other)
        where=np.searchsorted(codes,wanted)
        if (where==len(codes)).any() or (codes[where]!=wanted).any() or (multiplicity[where]!=1).any():
            raise ValueError('Each fine boundary edge must have exactly one background neighbor')
    # Weld each fine grid to the exact background boundary indices. Independent
    # hfield display geoms have vertical sides/base and float-transformed edges;
    # a ray through a tiny seam can otherwise hit that base instead of the top.
    chunks=[vertices];triangles=[faces];next_vertex=len(vertices)
    for index,center in enumerate(bank['centers']):
        z=bank['normalized'][index]*bank['height_range'][index]+bank['zmin'][index]+bank['altitudes'][index]
        interior=np.column_stack((fx[~edge]+center[0],fy[~edge]+center[1],z[~edge]))
        ids=np.empty(fx.shape,dtype=np.int32)
        ids[edge]=start+index*int(edge.sum())+np.arange(int(edge.sum()))
        ids[~edge]=np.arange(next_vertex,next_vertex+len(interior));next_vertex+=len(interior)
        a=ids[:-1,:-1].ravel();b=ids[:-1,1:].ravel();c=ids[1:,:-1].ravel();d=ids[1:,1:].ravel()
        triangles.extend((np.stack((a,b,d),-1),np.stack((a,d,c),-1)))
        chunks.append(interior)
    vertices=np.concatenate(chunks);faces=np.concatenate(triangles)
    spec.add_mesh(name='site20_full_view',uservert=vertices.astype(np.float32).ravel().tolist(),
                  userface=faces.ravel().tolist(),maxhullvert=4,smoothnormal=False)
    parent.add_geom(name='site20_full_visual',type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname='site20_full_view',pos=[-float(bank['centers'][0,0]),
                    -float(bank['centers'][0,1]),-float(bank['altitudes'][0])],
                    mass=0,contype=0,conaffinity=0,rgba=[.42,.43,.45,1.])


def configure_native(spec,full_view=False):
    bank=bank_data();n=bank['normalized'].shape[1]
    for index,grid in enumerate(bank['normalized']):
        spec.add_hfield(name=f'site20_local_{index:02d}',size=[HALF,HALF,float(bank['height_range'][index]),1.],
                        nrow=n,ncol=n,userdata=grid.ravel().tolist())
    old=next(g for g in spec.geoms if g.name=='terrain')
    # MJWarp deliberately never refreshes world-welded static geom_xpos after
    # make_data, even when geom_pos is a per-world randomized model field.
    # A stationary native mocap parent opts into native kinematics instead of
    # manually patching derived collision positions. No dynamic terrain DOFs.
    parent=spec.worldbody.add_body(name='site20_ground',mocap=True)
    old.name='site20_replaced_plane'
    geom=parent.add_geom(name='terrain',type=mujoco.mjtGeom.mjGEOM_HFIELD,
                         hfieldname='site20_local_00',group=old.group,
                         condim=old.condim,solref=old.solref,solimp=old.solimp)
    spec.delete(old)
    geom.pos=[0,0,float(bank['zmin'][0])];geom.quat=[1,0,0,0]
    geom.contype=1;geom.conaffinity=1;geom.friction=[1.,.01,.001]
    geom.rgba=[.42,.43,.45,1.]
    if full_view:
        add_full_view(spec,parent,bank)
        geom.rgba=[.42,.43,.45,0.]
    print(f'[Local Site20] {COUNT} shared native hfields, 16m patches, {STEP:g}m sampling; '
          f'5m source DEM + procedural relief; seed={lunar_details.settings().seed}; '
          f'max patch span={bank["height_range"].max():.3f}m',flush=True)


class LocalSurface(Surface):
    def __init__(self,env):
        bank=bank_data();self.meta=bank;self.half=HALF;self.step=STEP
        self.grid=torch.tensor(bank['normalized'],device=env.device)
        self.zmin=torch.tensor(bank['zmin'],device=env.device,dtype=torch.float32)
        self.height_range=torch.tensor(bank['height_range'],device=env.device,dtype=torch.float32)
        self.spawn_xy=torch.tensor(bank['spawn_xy'],device=env.device)
        self.spawn_counts=torch.tensor(bank['spawn_counts'],device=env.device,dtype=torch.long)
        self.spawnable_ids=(self.spawn_counts>0).nonzero(as_tuple=False).flatten()
        self.patch_ids=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        axis=torch.linspace(-.5,.5,5,device=env.device)
        self.scan_points=torch.cartesian_prod(axis,axis)
        model=env.sim.mj_model;self.geom_id=model.geom('terrain').id
        sole,cover=sole_sampling(model)
        if np.linalg.norm(sole[:,:2],axis=1).max()>.5:raise ValueError('Rest soles exceed certified spawn envelope')
        self.sole_points=torch.tensor(sole,device=env.device,dtype=torch.float32)
        axis=np.linspace(-HALF,HALF,self.grid.shape[1])
        mask=(axis[:-1]<2)&(axis[1:]>-2)
        z=bank['normalized']*bank['height_range'][:,None,None]+bank['zmin'][:,None,None]
        a=z[:,:-1,:-1];b=z[:,:-1,1:];c=z[:,1:,:-1];d=z[:,1:,1:]
        slopes=np.maximum(np.hypot(d-c,c-a),np.hypot(b-a,d-b))/STEP
        bounds=slopes[:,mask,:][:,:,mask].max((1,2))
        self.clearance_padding=torch.tensor(bounds*cover+2e-6,device=env.device,dtype=torch.float32)
        visual=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'site20_full_visual')
        self.visual_id=visual if visual>=0 else None
        self.visual_ids=None
        if self.visual_id is not None:
            ids=[self.visual_id]
            self.visual_ids=torch.tensor(ids,device=env.device,dtype=torch.long)
            initial=model.geom_pos[ids].copy()
            # Preserve native mesh compilation's centering/principal-axis
            # offsets; translate compiled poses, never reimplement mesh frames.
            shifts=np.column_stack((bank['centers'][0]-bank['centers'],bank['altitudes'][0]-bank['altitudes']))
            offsets=initial[None,:,:]+shifts[:,None,:]
            self.visual_offsets=torch.tensor(offsets,device=env.device,dtype=torch.float32)
        self.native_ids=torch.tensor([model.hfield(f'site20_local_{i:02d}').id for i in range(COUNT)],device=env.device,dtype=torch.int32)
        if model.nhfield!=COUNT or model.geom_type[self.geom_id]!=int(mujoco.mjtGeom.mjGEOM_HFIELD):
            raise ValueError('Local DEM collider missing')
        np.testing.assert_allclose(model.geom_pos[self.geom_id],[0,0,bank['zmin'][0]],atol=1e-6)
        for i in range(COUNT):
            fid=int(self.native_ids[i]);adr=int(model.hfield_adr[fid]);n=bank['normalized'][i].size
            np.testing.assert_allclose(model.hfield_data[adr:adr+n],bank['normalized'][i].ravel(),atol=2e-7,rtol=0)
            np.testing.assert_allclose(model.hfield_size[fid],[HALF,HALF,bank['height_range'][i],1.],atol=1e-8)

    def height(self,xy,env_ids=None):
        patch=self.patch_ids if env_ids is None else self.patch_ids[env_ids]
        patch=patch.view((-1,)+(1,)*(xy.ndim-2))
        ij=((xy+HALF)/STEP).clamp(0,self.grid.shape[1]-1-1e-4)
        col=ij[...,0].floor().long();row=ij[...,1].floor().long()
        u=ij[...,0]-col;v=ij[...,1]-row
        z00=self.grid[patch,row,col];z10=self.grid[patch,row,col+1]
        z01=self.grid[patch,row+1,col];z11=self.grid[patch,row+1,col+1]
        a=z00+u*(z11-z01)+v*(z01-z00)
        b=z00+u*(z10-z00)+v*(z11-z10)
        return torch.where(v>=u,a,b)*self.height_range[patch]+self.zmin[patch]
