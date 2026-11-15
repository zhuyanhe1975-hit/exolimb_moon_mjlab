"""Convert local Site20 DEM to portable MuJoCo terrain data (no simulation).
Original 5m DEM spacing and metric elevations are preserved; crop is 2km square.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.ndimage import maximum_filter

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=Path('/home/yhzhu/myWorks/zbot_exoarm_moon/assets/lunar_sites/southpole_site20/dem.npy')

def build(source, output):
    dem=np.load(source,mmap_mode='r')
    cy,cx=np.array(dem.shape)//2
    # DEM rows run north to south; MuJoCo hfield rows run -Y to +Y.
    height=np.flipud(dem[cy-200:cy+201,cx-200:cx+201]).astype(np.float32).copy()
    if height.shape!=(401,401) or not np.isfinite(height).all():
        raise ValueError('Expected a finite 401x401 center crop')
    centre=float(height[200,200]);height-=centre
    zmin=float(height.min());span=float(height.max()-height.min())
    normalized=((height-zmin)/span).astype(np.float32)
    # Bound BOTH triangle slopes of each native hfield cell, not central differences.
    z00,z10,z01,z11=height[:-1,:-1],height[:-1,1:],height[1:,:-1],height[1:,1:]
    slope_a=np.hypot((z11-z01)/5.,(z01-z00)/5.)
    slope_b=np.hypot((z10-z00)/5.,(z11-z10)/5.)
    cell_slope=np.maximum(slope_a,slope_b)
    # Spawn at cell centres with jitter <=1m; a 5x5-cell neighbourhood protects
    # the rotated soles and height scan. Steeper ground remains in the collider.
    safe=maximum_filter(cell_slope,size=5,mode='constant',cval=np.inf)<=np.tan(np.deg2rad(15.))
    safe[:3]=False;safe[-3:]=False;safe[:,:3]=False;safe[:,-3:]=False
    row,col=np.nonzero(safe)
    if len(row)==0: raise ValueError('No spawn cells satisfy the 15 degree bound')
    spawn_xy=np.column_stack((-1000+(col+.5)*5,-1000+(row+.5)*5)).astype(np.float32)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'heightfield.npz',normalized=normalized,spawn_xy=spawn_xy)
    report={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'site':'Site20 / Leibnitz beta plateau','source_resolution_m':5.,'shape':[401,401],
            'half_extent_m':1000.,'zmin_m':zmin,'height_range_m':span,'removed_centre_elevation_m':centre,
            'row_direction':'+Y','column_direction':'+X','triangle_diagonal':'(row,col) to (row+1,col+1)',
            'spawn_cells':len(row),'spawn_max_slope_deg':15.,'spawn_jitter_m':1.,
            'note':'Real DEM crop, no vertical exaggeration, no synthesized sub-5m rocks. Slopes >15deg are not spawn centres but remain collidable.'}
    report['asset_sha256']=hashlib.sha256((output/'heightfield.npz').read_bytes()).hexdigest()
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,default=ROOT/'assets/site20')
    args=p.parse_args();build(args.source,args.output)
