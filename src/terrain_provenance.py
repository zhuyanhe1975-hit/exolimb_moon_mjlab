"""Verify the actual generated terrain before loading a lunar policy."""
import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS=('centers','altitudes','zmin','height_range','normalized','spawn_xy','spawn_counts')


def add_playback_options(parser):
    """User-authorized automatic terrain transfer applies to playback only."""
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--terrain-transfer',dest='terrain_transfer',action='store_true',
                       help='Allow playback on current terrain when the snapshot is missing/different (default).')
    group.add_argument('--strict-terrain',dest='terrain_transfer',action='store_false',
                       help='Require a verified saved terrain; disable automatic terrain transfer.')
    parser.set_defaults(terrain_transfer=True)


def fingerprint():
    import local_lunar_terrain as terrain
    bank=terrain.bank_data();digest=hashlib.sha256()
    digest.update(terrain.REVISION.encode());digest.update(bank['detail_metadata']['sha256'].encode())
    digest.update(json.dumps(bank['source'],sort_keys=True).encode())
    for name in FIELDS:
        array=np.ascontiguousarray(bank[name]);digest.update(name.encode())
        digest.update(str((array.shape,array.dtype.str)).encode());digest.update(array.tobytes())
    for name in ('lunar_details.py','local_lunar_terrain.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def save(directory):
    import local_lunar_terrain as terrain
    destination=Path(directory);destination.mkdir(parents=True,exist_ok=True)
    bank=terrain.bank_data();archive=destination/'bank.npz'
    np.savez_compressed(archive,**{key:bank[key] for key in FIELDS})
    meta={'physics_revision':terrain.REVISION,'source':bank['source'],'detail':bank['detail_metadata'],
          'fingerprint':fingerprint(),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (destination/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


def verify(checkpoint,allow_transfer=False):
    source=Path(checkpoint).resolve();directory=source.parent/'terrain'
    reason=None
    try:
        meta=json.loads((directory/'metadata.json').read_text())
        if meta.get('fingerprint')!=fingerprint():reason='terrain configuration/generator/bank fingerprint differs'
        elif meta.get('archive_sha256')!=hashlib.sha256((directory/'bank.npz').read_bytes()).hexdigest():
            reason='saved terrain archive is missing or modified'
    except (OSError,ValueError):reason='verified terrain snapshot is missing or unreadable'
    if reason:
        if not allow_transfer:
            raise ValueError(f'{reason}: {source}. Use --terrain-transfer explicitly to test this policy on the current terrain; this is not an exact replay.')
        print(f'[Terrain transfer] {reason}; explicitly using CURRENT terrain; not exact replay',flush=True)
    else:print(f'[Terrain verified] {directory} | {meta["fingerprint"]}',flush=True)
