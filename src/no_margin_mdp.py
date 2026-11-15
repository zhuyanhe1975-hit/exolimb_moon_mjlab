"""Raw standing state without support margins or LIPM capture-point features."""
import torch
import balance_mdp as base
from support_mdp import phase_contact_data

# The legacy 145-column bundle puts LIPM prediction XY at columns 95:97,
# immediately after gravity derivatives and before the applied action vector.
# Drop these two columns as well as the entire six-column support extension.
OBS_DIM=143


def without_prediction(state):
    return torch.cat((state[:,:95],state[:,97:base.OBS_DIM]),-1)


def observation(env):
    return without_prediction(base.observation(env))


def guarded_observation(env):
    from standing_diagnostics import guarded_observation as guard
    return guard(env,observation_fn=observation,obs_dim=OBS_DIM)


def airborne(env):
    # Preserve the native loaded-contact definition, without computing a hull,
    # COM projection, capture point or margin just to detect missing contact.
    contact=phase_contact_data(env,'reward')
    valid=(contact.found>0)&(contact.force.norm(dim=-1)>.5)&(contact.dist<=.002)
    valid&=torch.isfinite(contact.pos).all(-1)
    return (~valid.any(-1)).float()
