"""Portable inference contract. Importing this module never starts a simulator."""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description='MJLab 146-observation/23-axis standing policy -> Isaac Sim 6.1 PhysX inference')
    p.add_argument('--task', choices=('stand-moon', 'stand-plane'), default='stand-moon')
    p.add_argument('--checkpoint-file', type=Path)
    p.add_argument('--experiment-name')
    p.add_argument('--terrain', choices=('auto', 'trained', 'moon', 'current', 'flat'), default='auto',
                   help='moon: current lunar terrain; auto: verified saved bank if available, otherwise moon; current is a legacy alias')
    from terrain_provenance import add_playback_options
    add_playback_options(p)
    p.add_argument('--patch', type=int, default=0)
    p.add_argument('--yaw', type=float, default=0., help='Initial heading in degrees')
    p.add_argument('--agent', choices=('trained', 'zero'), default='trained')
    p.add_argument('--num-envs','--num_envs',dest='num_envs',type=int,default=1,
                   help='Independent humans sharing ONE 16m moon patch (1..64); random safe XY/yaw')
    p.add_argument('--seed',type=int,help='Optional reproducible multi-human placement seed; default: new random seed each launch')
    p.add_argument('--follow',action=argparse.BooleanOptionalAction,default=False,
                   help='Start with position-only camera following; backslash toggles, comma/period selects human')
    p.add_argument('--seconds', type=float, default=60.)
    p.add_argument('--episode-seconds', type=float, default=10.)
    p.add_argument('--physics-substeps',type=int,default=4,
                   help='Subdivide each source physics interval for explicit PD (default: 480Hz physics, 60Hz policy)')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--render-every', '--render_every', type=int, default=None,
                   help='Render every N policy steps (default: 2 for multiple humans, 1 for one); physics and policy still run every step')
    p.add_argument('--verbose-init', action='store_true', help='Show MJLab/Isaac initialization output')
    p.add_argument('--realtime', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--skin', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--prepare-only', action='store_true', help='CPU asset/policy export only; do not start Kit or physics')
    p.add_argument('--bundle', type=Path, default=ROOT/'outputs/sim2sim_isaacsim/manual')
    p.add_argument('--report', type=Path)
    return p


def arguments(argv=None):
    p = parser(); a = p.parse_args(argv)
    if a.seconds <= 0 or a.episode_seconds <= 0 or not np.isfinite([a.seconds, a.episode_seconds, a.yaw]).all():
        p.error('seconds and episode-seconds must be positive and yaw must be finite')
    if a.patch < 0: p.error('patch must be nonnegative')
    if not 1<=a.num_envs<=64:p.error('num-envs must be between 1 and 64')
    if a.seed is not None and a.seed<0:p.error('seed must be nonnegative')
    if a.physics_substeps < 1:p.error('physics-substeps must be positive')
    if a.render_every is None:a.render_every = 2 if a.num_envs > 1 else 1
    if a.render_every < 1:p.error('render-every must be positive')
    if a.terrain in ('current','moon') and not a.terrain_transfer:
        p.error('moon/current terrain conflicts with --strict-terrain')
    a.bundle = a.bundle.resolve()
    return a


def resolve_terrain_mode(requested,task,checkpoint,allow_transfer):
    if requested=='moon':return 'current','explicit moon terrain; CURRENT lunar transfer, not exact replay'
    if requested!='auto':return requested,'explicit terrain mode'
    if task=='stand-plane':return 'flat','source task uses flat ground'
    directory=Path(checkpoint).parent/'terrain'
    try:
        meta=json.loads((directory/'metadata.json').read_text())
        if 'groundsync1' not in meta['physics_revision']:
            raise ValueError('saved terrain predates the GPU ground-position correction')
        if hashlib.sha256((directory/'bank.npz').read_bytes()).hexdigest()!=meta['archive_sha256']:
            raise ValueError('saved terrain archive hash mismatch')
        if not meta['fingerprint']:raise ValueError('saved terrain fingerprint is missing')
        return 'trained','verified saved terrain snapshot'
    except (OSError,ValueError,KeyError,TypeError) as error:
        reason=f'verified saved terrain unavailable: {error}'
        if not allow_transfer:
            raise ValueError(f'{reason}. Strict terrain mode refuses automatic transfer.') from error
        return 'current',reason+'; automatic CURRENT terrain transfer, not exact replay'


def load_manifest(directory):
    return json.loads((Path(directory)/'manifest.json').read_text())


def timing(manifest,substeps):
    """Keep policy timing while resolving the target's explicit torque feedback.

    Source implicitfast includes velocity-dependent PD damping in the solve;
    the target effort API evaluates it before each solve. Equal 120Hz outer
    steps are therefore not the same integrator. Do not retune the policy gains
    to hide that mismatch. Read state/recompute force on every target substep.
    """
    if not isinstance(substeps,int) or substeps<1:raise ValueError('substeps must be a positive integer')
    return float(manifest['physics_dt'])/substeps, int(manifest['decimation'])*substeps, float(manifest['physics_dt'])*int(manifest['decimation'])


def height(xy, grid, half):
    """Same SW->NE triangle interpolation as the native training hfield.

    This is the policy's existing height lookup, not a new contact estimator.
    PhysX's collision mesh is exported from exactly these triangles.
    """
    xy = np.asarray(xy)
    if grid is None: return np.zeros(xy.shape[:-1])
    if not np.isfinite(xy).all() or np.any(np.abs(xy) > half):
        raise ValueError('Policy terrain query outside its finite collision patch')
    ij = np.minimum((xy+half)*(grid.shape[0]-1)/(2*half), grid.shape[0]-1-1e-4)
    col, row = np.floor(ij[...,0]).astype(int), np.floor(ij[...,1]).astype(int)
    u, v = ij[...,0]-col, ij[...,1]-row
    z00, z10 = grid[row,col], grid[row,col+1]
    z01, z11 = grid[row+1,col], grid[row+1,col+1]
    return np.where(v >= u, z00+u*(z11-z01)+v*(z01-z00), z00+u*(z10-z00)+v*(z11-z10))


def triangles(grid, half):
    n = len(grid)
    x, y = np.meshgrid(np.linspace(-half, half, n), np.linspace(-half, half, n))
    points = np.stack((x, y, grid), -1).reshape(-1,3)
    base = (np.arange(n-1)[:,None]*n+np.arange(n-1)[None,:]).ravel()
    faces = np.concatenate((np.stack((base,base+1,base+n+1),-1), np.stack((base,base+n+1,base+n),-1)))
    return points.astype(np.float32), faces.astype(np.int32)


def action_torque(action, q, qd, manifest):
    """Actor order is JOINT order; actuator gains are mapped by trnid on export.

    Native MuJoCo clips the PD actuator torque before adding passive joint
    damping. PhysX effort mode has no second position/velocity drive enabled.
    """
    action = np.clip(action, -manifest['clip_actions'], manifest['clip_actions'])
    target = np.clip(action*manifest['action_scale']+np.array(manifest['q_default']),
                     np.array(manifest['target_limits'])[:,0], np.array(manifest['target_limits'])[:,1])
    pd = np.clip(np.array(manifest['kp'])*(target-q)-np.array(manifest['kd'])*qd,
                 np.array(manifest['effort_limits'])[:,0], np.array(manifest['effort_limits'])[:,1])
    return pd-np.array(manifest['joint_damping'])*qd, action, pd
