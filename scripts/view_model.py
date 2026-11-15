"""Display the baked Wangyu visual without stepping physics."""
from pathlib import Path
import time
import mujoco
import mujoco.viewer

path = Path(__file__).resolve().parents[1] / 'assets/wangyu_visual/scene.xml'
model = mujoco.MjModel.from_xml_path(str(path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = (0, 0, .9)
    viewer.cam.distance = 3.3
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -10
    while viewer.is_running():
        viewer.sync()
        time.sleep(1 / 60)
