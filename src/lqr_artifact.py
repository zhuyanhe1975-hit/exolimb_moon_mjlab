"""Versioned, non-pickle LQR artifact. Importing this module never simulates."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / 'outputs/lqr/stand_lqr_v1.npz'
SCHEMA = 1
NQ, NV, NU = 30, 29, 23
# Flat ground has translation-XY/yaw gauge freedoms. Do not introduce a station
# position objective: retain height, roll/pitch, all hinges and all velocities.
KEEP = np.array([2, 3, 4, *range(6, NV), *range(NV, 2*NV)], dtype=np.int64)
STATE_DIM = len(KEEP)


def source_digest():
    digest = hashlib.sha256()
    for name in ('assets/wangyu_physics/wangyu.xml', 'src/wangyu23.py',
                 'src/stand_task.py', 'src/lqr_artifact.py', 'src/lqr_action.py',
                 'scripts/prepare_stand_lqr.py', 'src/lqr_task.py'):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_artifact(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'LQR gain missing: {path}. Run ./train.sh stand-lqr '
                                '(prepares it once), or scripts/prepare_stand_lqr.py.')
    with np.load(path, allow_pickle=False) as archive:
        out = {name: archive[name].copy() for name in archive.files}
    meta = json.loads(str(out.pop('metadata')))
    if meta.get('schema') != SCHEMA or meta.get('source_sha256') != source_digest():
        raise ValueError('LQR schema/physics source mismatch; explicitly rebuild the gain. '
                         'Do not reuse an old checkpoint with a newly generated controller.')
    expected = {'K': (NU, STATE_DIM), 'qpos0': (NQ,), 'ctrl0': (NU,),
                'keep': (STATE_DIM,), 'joint_names': (NU,), 'actuator_joints': (NU,),
                'kp': (NU,), 'kd': (NU,), 'force_limit': (NU,),
                'joint_range': (NU, 2), 'body_mass': (14,)}
    for key, shape in expected.items():
        if key not in out or out[key].shape != shape:
            raise ValueError(f'Invalid LQR artifact field {key}, expected {shape}')
        if out[key].dtype.kind in 'fci' and not np.isfinite(out[key]).all():
            raise ValueError(f'Nonfinite LQR field: {key}')
    if not np.array_equal(out['keep'], KEEP):
        raise ValueError('Unsupported LQR tangent-state ordering')
    if not np.isclose(np.linalg.norm(out['qpos0'][3:7]),1.,atol=1e-6):
        raise ValueError('Invalid reference quaternion')
    if len(set(out['joint_names'])) != NU or set(out['joint_names']) != set(out['actuator_joints']):
        raise ValueError('Invalid LQR actuator/joint mapping')
    if (not meta.get('validated') or not 0 <= meta.get('spectral_radius', 2.) < 1.
            or not np.isclose(meta.get('gravity', 0.), 9.81/6)
            or not np.isclose(meta.get('physics_dt', 0.), 1/120)
            or meta.get('decimation') != 2):
        raise ValueError('LQR artifact lacks successful equilibrium/closed-loop checks')
    return out, meta, file_digest(path)
