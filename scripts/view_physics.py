"""Native Wangyu viewer. Space toggles passive simulation; R resets."""
import argparse
import time
from pathlib import Path
import mujoco
import mujoco.viewer
from wangyu23 import restore_23_axes

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--simulate',action='store_true',help='Start passive physics immediately (no standing controller).')
parser.add_argument('--proxy',action='store_true',help='Show native physics primitives instead of the suit.')
parser.add_argument('--skin',action='store_true',help='Show the bound suit (default).')
args=parser.parse_args()
model=restore_23_axes(mujoco.MjSpec.from_file(str(Path(__file__).resolve().parents[1]/'assets/wangyu_skinned.xml'))).compile()
for i in range(model.ngeom):
    if model.geom_bodyid[i]!=0: model.geom_group[i]=3
data=mujoco.MjData(model)
mujoco.mj_forward(model,data)
state={'running':args.simulate,'reset':False,'proxy':args.proxy and not args.skin}
def key(code):
    if code==32: state['running']=not state['running']
    if code in (ord('R'),ord('r')): state['reset']=True
    if code in (ord('P'),ord('p')): state['proxy']=not state['proxy']
print('Wangyu native physics | 23 driven axes | head/wrists fixed | 126.8 kg engineering estimate',flush=True)
print('Space: run/pause passive physics; R: reset; P: suit/physics primitives. No policy or standing controller.',flush=True)
with mujoco.viewer.launch_passive(model,data,key_callback=key) as viewer:
    viewer.cam.lookat[:]=(0,0,.95)
    viewer.cam.distance=3.5
    viewer.cam.azimuth=135
    viewer.cam.elevation=-12
    while viewer.is_running():
        started=time.monotonic()
        with viewer.lock():
            viewer.opt.geomgroup[3]=state['proxy']
            viewer.opt.skingroup[1]=not state['proxy']
            if state['reset']:
                mujoco.mj_resetData(model,data)
                mujoco.mj_forward(model,data)
                state.update(reset=False,running=False)
            if state['running']:
                for _ in range(2): mujoco.mj_step(model,data)
        viewer.sync()
        time.sleep(max(0,2*model.opt.timestep-(time.monotonic()-started)))
