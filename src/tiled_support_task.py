"""v11 contact experiment: four exact box tiles per sole, unchanged inertia.

Heightfield CCD can yield one point for a box within one terrain prism. Split
only the collision shape so multiple independent native pairs can carry load.
No synthetic contact points, extra mass, support forces or viewer-only fixes.
"""
from copy import deepcopy
import os
import numpy as np
import mujoco
import support_task as base
import balance_task as physical_base
from mjlab.sensor import ContactSensorCfg,ContactMatch
from mjlab.tasks.registry import register_mjlab_task
from stand_runner import StandRunner

TASK_ID='Wangyu-Moon-Stand-Support-Contact-v11'
EXPERIMENT=os.environ.get('WANGYU_TRAIN_EXPERIMENT','wangyu_moon_stand_support_contact_v11_ccd1e3_contactsync1')
OBS_DIM=base.OBS_DIM
SPAWN_MARGIN,STATION_RADIUS=base.SPAWN_MARGIN,base.STATION_RADIUS
PUSH_PARAMS=base.PUSH_PARAMS
REWARD_WEIGHTS=dict(base.REWARD_WEIGHTS)
CCD_TOLERANCE=.001


def tile_soles(spec):
    # The skin spec intentionally has robot/-prefixed material references and
    # cannot be compiled until scene attachment. Obtain inertia from the same
    # native physical asset without skin, as validated by the skin audit.
    source=physical_base.robot_spec().compile()
    properties=('contype','conaffinity','condim','friction','solref','solimp','solmix',
                'margin','gap','priority','group','rgba')
    for side in ('Left','Right'):
        body=next(b for b in spec.bodies if b.name==side+'Foot');bid=source.body(side+'Foot').id
        # MuJoCo computes the baseline inertia. Freeze its exact native result;
        # collider subdivision must not change mass, COM or inertia tensor.
        body.mass=float(source.body_mass[bid]);body.ipos=source.body_ipos[bid].copy()
        body.iquat=source.body_iquat[bid].copy();body.inertia=source.body_inertia[bid].copy()
        body.explicitinertial=True
        # Compiled name lookup can keep old geom indices after inserting tiles
        # into the preceding body; resolve the mutable spec by element name.
        geom=next(g for g in spec.geoms if g.name==side.lower()+'_sole')
        size=np.array(geom.size);position=np.array(geom.pos);quat=np.array(geom.quat)
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,quat);rotation=rotation.reshape(3,3)
        attributes={key:deepcopy(getattr(geom,key)) for key in properties}
        for index,(x,y) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
            tile=geom if index==0 else body.add_geom(name=side.lower()+f'_sole_tile_{index}')
            tile.type=mujoco.mjtGeom.mjGEOM_BOX
            tile.size=[size[0]/2,size[1]/2,size[2]]
            tile.pos=position+rotation@np.array([x*size[0]/2,y*size[1]/2,0.])
            tile.quat=quat
            for key,value in attributes.items():setattr(tile,key,value)
            tile.mass=0.
    return spec


def robot_spec():
    return tile_soles(physical_base.robot_spec())


def skin_spec():
    return tile_soles(physical_base.skin_spec())


def configure_contact_scene(spec):
    # Native absolute distance tolerance. At ~1km coordinates float32 spacing
    # is ~0.1mm: 1um convergence is below that resolution. A 1mm tolerance is
    # a diagnostic profile, not proof that all heightfield CCD bugs are fixed.
    spec.option.ccd_tolerance=CCD_TOLERANCE
    print(f'[CCD contact profile] native ccd_tolerance={CCD_TOLERANCE:g}m; four-tile sole; numerical guard retained',flush=True)


def configure_contact_view(spec):
    physical_base.configure_balance_view(spec)
    configure_contact_scene(spec)


def terrain_stand_cfg(play=False):
    cfg=base.terrain_stand_cfg(play=play)
    cfg.scene.spec_fn=configure_contact_view if play else configure_contact_scene
    cfg.scene.entities['robot'].spec_fn=skin_spec if play else robot_spec
    # Aggregate actual tile contacts per foot; retain the same 8 observation slots.
    cfg.scene.sensors=tuple(ContactSensorCfg(name='support_contacts',
        primary=ContactMatch(mode='body',pattern=('LeftFoot','RightFoot'),entity='robot'),
        secondary=ContactMatch(mode='geom',pattern='site20_surface'),
        fields=('found','force','dist','pos'),reduce='maxforce',num_slots=4)
        if sensor.name=='support_contacts' else sensor for sensor in cfg.scene.sensors)
    return cfg


terrain_stand_rl=deepcopy(base.terrain_stand_rl)
terrain_stand_rl.experiment_name=EXPERIMENT
register_mjlab_task(TASK_ID,terrain_stand_cfg(),terrain_stand_cfg(play=True),terrain_stand_rl,runner_cls=StandRunner)
