"""Site20 native hfield, exact triangle sampling and GPU random spawn support."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import os
import numpy as np
import mujoco
import torch
from tensor_graph import FixedTensorGraph
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput, TerrainGeometry

ASSET=Path(__file__).resolve().parents[1]/'assets/site20'

@lru_cache(maxsize=1)
def terrain_data():
    metadata=json.loads((ASSET/'metadata.json').read_text())
    with np.load(ASSET/'heightfield.npz') as archive:
        return metadata,archive['normalized'].copy(),archive['spawn_xy'].copy()


@dataclass
class Site20TerrainCfg(SubTerrainCfg):
    def function(self,difficulty,spec,rng):
        del difficulty,rng
        meta,normalized,_=terrain_data()
        half=meta['half_extent_m']
        if self.size!=(2*half,2*half):
            raise ValueError('Site20 metric size must not be rescaled')
        field=spec.add_hfield(name='site20_field',size=[half,half,meta['height_range_m'],1.],
                              nrow=normalized.shape[0],ncol=normalized.shape[1],
                              userdata=normalized.ravel().tolist())
        geom=spec.body('terrain').add_geom(name='site20_surface',type=mujoco.mjtGeom.mjGEOM_HFIELD,
                                           hfieldname=field.name,pos=[half,half,meta['zmin_m']],
                                           contype=1,conaffinity=1,friction=[1.,.01,.001],rgba=[.42,.43,.45,1.])
        # TerrainGenerator shifts this single patch by (-half,-half,0).
        origin=np.array([half,half,float(normalized[200,200]*meta['height_range_m']+meta['zmin_m'])])
        return TerrainOutput(origin=origin,geometries=[TerrainGeometry(geom=geom,hfield=field)])


class Surface:
    def __init__(self,device):
        self.meta,normalized,spawn=terrain_data()
        self.grid=torch.tensor(normalized,device=device,dtype=torch.float32)
        self.spawn_xy=torch.tensor(spawn,device=device,dtype=torch.float32)
        self.half=self.meta['half_extent_m']
        self.step=2*self.half/(self.grid.shape[0]-1)
        manifest=json.loads((ASSET.parent/'wangyu_physics/manifest.json').read_text())
        # Native rest frames retain world orientation; compute sole point geometry
        # from the saved manifest, without compiling or stepping any model.
        pelvis=np.array(manifest['bodies']['Pelvis']['neutral_origin_world_m'])
        points=[]
        for geom in manifest['geometries']:
            if not geom['name'].endswith('_sole'): continue
            origin=np.array(manifest['bodies'][geom['body']]['neutral_origin_world_m'])
            pos=np.fromstring(geom['pos'],sep=' ');size=np.fromstring(geom['size'],sep=' ')
            for x in np.linspace(-size[0],size[0],5):
                for y in np.linspace(-size[1],size[1],5):
                    points.append(origin+pos+np.array([x,y,-size[2]])-pelvis)
        if len(points)!=50: raise ValueError('Expected 25 bottom samples per native box sole')
        self.sole_points=torch.tensor(np.array(points),device=device,dtype=torch.float32)
        coordinates=torch.linspace(-.5,.5,5,device=device)
        self.scan_points=torch.cartesian_prod(coordinates,coordinates)

    def height(self,xy):
        """MuJoCo 3.11 hfield triangle diagonal: southwest to northeast.
        Read against engine_collision_convex.c:addPrismVert. Never bilinear.
        Outside points are clamped for finite observations; edge termination handles them.
        """
        ij=((xy+self.half)/self.step).clamp(0.,self.grid.shape[0]-1-1e-4)
        col=ij[...,0].floor().long();row=ij[...,1].floor().long()
        u=ij[...,0]-col;v=ij[...,1]-row
        z00=self.grid[row,col];z10=self.grid[row,col+1]
        z01=self.grid[row+1,col];z11=self.grid[row+1,col+1]
        a=z00+u*(z11-z01)+v*(z01-z00)
        b=z00+u*(z10-z00)+v*(z11-z10)
        return torch.where(v>=u,a,b)*self.meta['height_range_m']+self.meta['zmin_m']

    def scan(self,root,center_height=None):
        q=root[:,3:7];w,x,y,z=q.unbind(-1)
        c=1-2*(y*y+z*z);s=2*(w*z+x*y)
        norm=torch.sqrt(c*c+s*s).clamp_min(1e-6);c=c/norm;s=s/norm
        local=self.scan_points
        xx=root[:,None,0]+c[:,None]*local[None,:,0]-s[:,None]*local[None,:,1]
        yy=root[:,None,1]+s[:,None]*local[None,:,0]+c[:,None]*local[None,:,1]
        height=self.height(torch.stack((xx,yy),-1))
        center=self.height(root[:,:2]) if center_height is None else center_height
        return (5*(height-center[:,None])).clamp(-5.,5.)

    def root_height(self,xy):
        """Fixed-batch per-step query; reset's variable-size queries stay eager.

        No cross-step value cache: termination and observation can read different
        kinematic phases within a control step. Copy current coordinates each call.
        """
        if xy.is_cuda and os.environ.get('WANGYU_TENSOR_GRAPH','1')!='0':
            if not hasattr(self,'_root_height_graph'):
                self._root_height_graph=FixedTensorGraph(self.height,xy)
            return self._root_height_graph(xy)
        return self.height(xy)


class FlatSurface:
    """Exact z=0 native plane; no DEM reads, triangle queries or terrain scan."""
    def height(self,xy):
        return torch.zeros_like(xy[...,0])

    def root_height(self,xy):
        return self.height(xy)

    def scan(self,root,center_height=None):
        return root.new_zeros((root.shape[0],25))


def surface(env):
    if not hasattr(env,'_lunar_surface'):
        terrain=getattr(env.cfg.scene,'terrain',None)
        env._lunar_surface=FlatSurface() if getattr(terrain,'terrain_type',None)=='plane' else Surface(env.device)
    return env._lunar_surface
