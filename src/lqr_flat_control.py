"""Delay-free, bumpless flat-ground LQR authority, using calibrated PD feedforward."""
import torch
from lqr_action import apply_residual


def equilibrium_commands(dx,K,ctrl0,authority,residual,lower,upper,
                         residual_scale=.05,fallback_scale=.25):
    # C1 transition: no temporal filter or added control state/delay. Crucially,
    # reducing feedback authority must NOT interpolate ctrl0 toward source q=0.
    a=authority.clamp(0.,1.)
    a=a.square()*(3.-2.*a)
    requested=ctrl0-a[:,None]*(dx@K.T)
    base=requested.clamp(lower,upper)
    scale=(residual_scale+(1.-a)*(fallback_scale-residual_scale))[:,None]
    final,clipped=apply_residual(base,residual,lower,upper,scale)
    clipped=torch.maximum(clipped,(requested-base).abs().gt(1e-6).any(-1).float())
    return base,final,a,clipped
