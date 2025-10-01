import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the Unitree G1 model
model = mujoco.MjModel.from_xml_path('xmls/unitree_g1/scene.xml')
data = mujoco.MjData(model)

HZ = 50
FLOATING_XYZ = np.array([0.0, 0.0, 0.8])
FLOATING_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
FLOATING_FRAME = np.hstack([FLOATING_XYZ, FLOATING_QUAT])
PERIOD = 2

sim_start = time.time()
# Simple walking-ish gait: sinusoidal hip/knee swings
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        time_start = data.time
        real_time  = time.time()
        while data.time - time_start < 1.0 / HZ:
            t = data.time

            mujoco.mj_step(model, data)
            data.qpos[:7] = FLOATING_FRAME.copy()
            data.qpos[23] = np.pi / 2
            data.qpos[24] = np.pi / 2
            data.qpos[25] = 0.3 * np.sin(2 * np.pi * 1 / PERIOD * t)
            data.qvel[:]  = np.zeros(model.nv)
            data.qacc[:]  = np.zeros(model.nv)

        viewer.sync()
        left_over_time = time.time() - real_time
        time.sleep(max(1.0 / HZ - left_over_time, 0.0))

viewer.close()