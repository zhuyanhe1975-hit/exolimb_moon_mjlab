"""Bind each residual policy to its exact controller; preserve StandLogger."""
from pathlib import Path

import torch
from lqr_action import term
from lqr_artifact import file_digest
from stand_runner import StandRunner


class LqrRunner(StandRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        policy = self.alg.get_policy()
        linear = [item for item in policy.mlp.modules() if isinstance(item,torch.nn.Linear)]
        if not linear or linear[-1].out_features != 23:
            raise ValueError('Expected a 23-dimensional Gaussian residual mean head')
        with torch.no_grad():
            linear[-1].weight.zero_()
            linear[-1].bias.zero_()
        controller = term(self.env.unwrapped)
        self.controller_bytes = controller.artifact_path.read_bytes()
        self.controller_sha = controller.sha256
        if file_digest(controller.artifact_path) != self.controller_sha:
            raise RuntimeError('LQR file changed during runner construction')

    def save(self, path, infos=None):
        destination = Path(path).parent/'lqr/controller.npz'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if file_digest(destination) != self.controller_sha:
                raise RuntimeError('Refusing to overwrite a run with a different LQR controller')
        else:
            destination.write_bytes(self.controller_bytes)
        return super().save(path,infos={**(infos or {}),'lqr_sha256':self.controller_sha})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        snapshot = Path(path).parent/'lqr/controller.npz'
        if not snapshot.is_file() or file_digest(snapshot) != self.controller_sha:
            raise ValueError('Checkpoint LQR snapshot missing or differs from the active controller. '
                             'Set WANGYU_LQR_FILE to RUN/lqr/controller.npz before resume.')
        return super().load(path,load_cfg=load_cfg,strict=strict,map_location=map_location)
