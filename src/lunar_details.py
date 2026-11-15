"""Deterministic procedural sub-DEM relief, inspired by OmniLRS layering.

References: local OmniLRS large_scale_terrain/crater_distribution.py (Poisson
sampling + rejection), crater_generation.py (deformed radial crater profiles),
and terrain_generation.py (multiple roughness scales). Independently implemented
NumPy profiles; no OmniLRS source/assets or simulator dependencies are imported.
This is synthetic relief, not measured Site20 features or a regolith model.
"""
from dataclasses import dataclass,asdict
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np

CONFIG=Path(__file__).resolve().parents[1]/'assets/site20/detail_config.json'


@dataclass(frozen=True)
class DetailConfig:
    enabled:bool=True
    seed:int=20261114
    resolution_m:float=.1
    roughness_amplitude_m:float=.008
    crater_density_m2:float=.12
    crater_radius_m:tuple=(.3,1.)
    crater_depth_radius_ratio:float=.075
    rim_depth_ratio:float=.3
    min_rim_sigma_cells:float=1.5
    mound_density_m2:float=.18
    mound_radius_m:tuple=(.4,.6)
    mound_height_m:tuple=(.01,.035)

    def __post_init__(self):
        scalars=[self.resolution_m,
                 self.roughness_amplitude_m,self.crater_density_m2,self.crater_depth_radius_ratio,
                 self.rim_depth_ratio,self.mound_density_m2]
        if not np.isfinite(scalars).all() or min(scalars)<0:
            raise ValueError('Detail parameters must be finite and nonnegative')
        if type(self.seed) is not int or self.seed<0 or type(self.enabled) is not bool:
            raise ValueError('seed must be a nonnegative integer; enabled must be boolean')
        if self.resolution_m!=.1:
            raise ValueError('Validated native detail sampling is 0.1m; re-audit before changing it')
        if not np.isfinite(self.min_rim_sigma_cells) or self.min_rim_sigma_cells<1.5:
            raise ValueError('Rim sigma must span at least 1.5 grid intervals')
        if min(self.mound_radius_m)*.8<3*self.resolution_m:
            raise ValueError('Mound minor diameter must span at least 6 grid intervals')
        for interval in (self.crater_radius_m,self.mound_radius_m,self.mound_height_m):
            if len(interval)!=2 or not np.isfinite(interval).all() or not 0<interval[0]<=interval[1]:
                raise ValueError('Invalid detail interval')


@lru_cache(maxsize=1)
def settings():
    return DetailConfig(**json.loads(CONFIG.read_text()))


def metadata():
    config=asdict(settings())
    return {'kind':'procedural, not measured','reference':'OmniLRS layered DEM / crater / roughness approach',
            'configuration':config,'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}


@lru_cache(maxsize=2048)
def _cell_features(ix,iy,config_json):
    """Canonical feature metadata, independent of crop and request order."""
    cfg=DetailConfig(**json.loads(config_json))
    zig=lambda n:2*n if n>=0 else -2*n-1
    rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,zig(ix),zig(iy),3103]))
    craters=[];mounds=[]
    for _ in range(rng.poisson(100*cfg.crater_density_m2)):
        x,y=rng.uniform(0,10,2)+[10*ix,10*iy]
        craters.append((x,y,rng.uniform(*cfg.crater_radius_m),rng.uniform(0,2*np.pi),rng.random()))
    for _ in range(rng.poisson(100*cfg.mound_density_m2)):
        x,y=rng.uniform(0,10,2)+[10*ix,10*iy]
        mounds.append((x,y,rng.uniform(*cfg.mound_radius_m),rng.uniform(*cfg.mound_height_m)))
    return tuple(craters),tuple(mounds)


def generate(xy,center,config=None):
    """Global additive field sampled at local xy + source-map center.

    Padded canonical-cell metadata includes neighboring craters before
    priority rejection. Features cross cell/crop boundaries without a fade mask.
    """
    cfg=config or settings()
    if not cfg.enabled:return np.zeros(xy.shape[:-1],dtype=np.float64)
    world=np.asarray(xy,dtype=np.float64)+np.asarray(center,dtype=np.float64)
    if not world.size:return np.zeros(world.shape[:-1])
    if not np.isfinite(world).all():raise ValueError('Nonfinite global detail coordinates')
    flat=world.reshape(-1,2)
    if np.ptp(flat,axis=0).max()>32:
        # Sparse world-wide requests must not build one quadratic rejection
        # problem containing every crater between the furthest query points.
        cells,inverse=np.unique(np.floor(flat/10).astype(np.int64),axis=0,return_inverse=True)
        output=np.empty(len(flat))
        for index in range(len(cells)):
            selected=inverse==index
            output[selected]=generate(flat[selected],[0,0],cfg)
        return output.reshape(world.shape[:-1])
    x,y=world[...,0],world[...,1];height=np.zeros_like(x)
    rng=np.random.default_rng([cfg.seed,3103])
    for wavelength,weight in ((.4,.10),(.8,.20),(1.6,.30),(3.2,.40)):
        for _ in range(3):
            angle=rng.uniform(0,2*np.pi);phase=rng.uniform(0,2*np.pi)
            height+=cfg.roughness_amplitude_m*weight/3*np.sin(
                2*np.pi*(x*np.cos(angle)+y*np.sin(angle))/wavelength+phase)
    max_r=max(cfg.crater_radius_m)
    support=(max_r+4*max(.16*max_r,cfg.min_rim_sigma_cells*cfg.resolution_m))/.94
    rejection=1.4*max_r
    padding=max(support+rejection,max(cfg.mound_radius_m))
    low=world.reshape(-1,2).min(0);high=world.reshape(-1,2).max(0)
    cell_low=np.floor((low-padding)/10).astype(int);cell_high=np.floor((high+padding)/10).astype(int)
    key=json.dumps(asdict(cfg),sort_keys=True)
    craters=[];mounds=[]
    for ix in range(cell_low[0],cell_high[0]+1):
        for iy in range(cell_low[1],cell_high[1]+1):
            a,b=_cell_features(ix,iy,key);craters.extend(a);mounds.extend(b)
    raw=np.asarray(craters,dtype=float).reshape(-1,5)
    for cx,cy,radius,phase,priority in raw:
        sigma=max(.16*radius,cfg.min_rim_sigma_cells*cfg.resolution_m)
        extent=(radius+4*sigma)/.94
        if cx+extent<low[0] or cx-extent>high[0] or cy+extent<low[1] or cy-extent>high[1]:continue
        close=np.hypot(raw[:,0]-cx,raw[:,1]-cy)<.7*(raw[:,2]+radius)
        if np.any(close&(raw[:,4]<priority)):continue
        theta=np.arctan2(y-cy,x-cx)
        rho=np.hypot(x-cx,y-cy)/radius*(1+.06*np.sin(3*theta+phase))
        depth=radius*cfg.crater_depth_radius_ratio
        bowl=-depth*np.maximum(1-rho*rho,0)**2
        rim=depth*cfg.rim_depth_ratio*np.exp(-.5*((rho*radius-radius)/sigma)**2)
        cutoff=np.clip((radius+4*sigma-rho*radius)/(2*sigma),0,1)
        height+=bowl+rim*cutoff*cutoff*(3-2*cutoff)
    for cx,cy,radius,amplitude in mounds:
        if cx+radius<low[0] or cx-radius>high[0] or cy+radius<low[1] or cy-radius>high[1]:continue
        rho2=((x-cx)/radius)**2+((y-cy)/(radius*.8))**2
        height+=amplitude*np.maximum(1-rho2,0)**2
    return height
