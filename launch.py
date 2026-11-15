import sys
from training_options import configure
sys.argv=configure(sys.argv)
import moon_task
import stand_task
import terrain_stand_task
import balance_v3_task
import balance_task
import stand_push_task
import stand_pose_free_task
import support_task
import flat_support_task
import flat_no_margin_task
import flat_torso_observation_task
import jump_task
import lunar_finetune_task
import tiled_support_task
import lqr_task
import lqr_flat_task
import lqr_flat_v6_task
if any(arg.startswith(('Wangyu-Moon-Stand','Wangyu-Moon-Hop')) for arg in sys.argv):
 import torch
 torch.set_num_threads(1)
if sys.argv[1]=='train':
 sys.argv=sys.argv[:1]+sys.argv[2:]
 from mjlab.scripts.train import main
else:
 sys.argv=sys.argv[:1]+sys.argv[2:]
 from mjlab.scripts.play import main
main()
