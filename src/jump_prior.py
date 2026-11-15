"""Strict actor-only warm start from the unmodified 146-input standing model."""
from pathlib import Path
import hashlib
import json
import os
import torch
from stand_runner import StandRunner


class JumpRunner(StandRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        prior=os.environ.get('WANGYU_JUMP_PRIOR')
        if prior:
            path=Path(prior).resolve()
            checkpoint=torch.load(path,map_location=self.device,weights_only=False)
            self.alg.get_policy().load_state_dict(checkpoint['actor_state_dict'],strict=True)
            print(f'[Standing prior] {path}; actor/normalizer/distribution strictly unchanged (146 inputs); fresh critic/optimizer/iteration',flush=True)
            if self.logger.log_dir:
                report={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'source_iteration':checkpoint.get('iter'),'source_inputs':146,'jump_inputs':146,
                        'loaded':['actor','actor_normalizer'],'fresh':['critic','optimizer','iteration'],
                        'method':'exact trainable standing actor warm start; no input expansion or phase'}
                (Path(self.logger.log_dir)/'standing_prior.json').write_text(json.dumps(report,indent=2)+'\n')
