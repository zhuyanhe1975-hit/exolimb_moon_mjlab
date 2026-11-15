"""Lightweight experiment naming and jump-height validation."""
import math

DEFAULT_JUMP_HEIGHT=.25
REWARD_REVISION='momentum_height_exp1'
MOMENTUM_EXPONENT=5.0
HEIGHT_EXPONENT=5.0


def validate_height(height):
    value=float(height)
    if not math.isfinite(value) or value<=0:
        raise ValueError('jump-height must be finite and positive (metres)')
    return value


def experiment_base(physics_revision,height):
    return f'wangyu_moon_hop_lunar_v2_{physics_revision}_{REWARD_REVISION}_km{MOMENTUM_EXPONENT:g}_kh{HEIGHT_EXPONENT:g}_h{validate_height(height):g}'
