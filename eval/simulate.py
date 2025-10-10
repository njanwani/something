import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

ARM_POS = 0.7 * (2 * np.random.random(6) - 1)

# Load the Unitree G1 model
path = Path('xmls/scene.xml')
model = mujoco.MjModel.from_xml_string(path.read_text())
data = mujoco.MjData(model)

def update_pos(t, _model, _data):
    _data.qpos[:7] = FLOATING_FRAME.copy()
    _data.qpos[23] = np.pi / 2
    _data.qpos[24] = np.pi / 2
    _data.qpos[25] = 0.3 * np.sin(2 * np.pi * 1 / PERIOD * t)
    _data.qpos[30] = -0.178
    _data.qpos[32] = 1.4
    _data.qvel[:]  = np.zeros(_model.nv)
    _data.qacc[:]  = np.zeros(_model.nv)
    _data.qpos[-6:] = ARM_POS.copy()
    data.ctrl = data.qpos[7:]
    return _data

HZ = 50
FLOATING_XYZ = np.array([0.0, 0.0, 0.8])
FLOATING_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
FLOATING_FRAME = np.hstack([FLOATING_XYZ, FLOATING_QUAT])
PERIOD = 1.5
data = update_pos(0.0, model, data)

sim_start = time.time()
# Simple walking-ish gait: sinusoidal hip/knee swings
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        time_start = data.time
        real_time  = time.time()
        while data.time - time_start < 1.0 / HZ:
            data = update_pos(data.time, model, data)
            mujoco.mj_step(model, data)
            

        viewer.sync()
        left_over_time = time.time() - real_time
        time.sleep(max(1.0 / HZ - left_over_time, 0.0))

viewer.close()