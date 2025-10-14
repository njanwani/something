import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from kinematics.chain import Chain
from kinematics.solver import imitate

# Load the Unitree G1 model
path = Path('xmls/generic_arm/arm6DOF.xml')
model = mujoco.MjModel.from_xml_string(path.read_text())
data = mujoco.MjData(model)

HZ = 50
# Set new local coordinates (relative to the parent body)

sim_start = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        time_start = data.time
        real_time  = time.time()
        while data.time - time_start < 1.0 / HZ:
            mujoco.mj_step(model, data)
            data.qpos = 0.0
            data.qpos[2] = np.pi / 2
            data.qvel = 0.0
            data.qacc = 0.0


        viewer.sync()
        left_over_time = time.time() - real_time
        time.sleep(max(1.0 / HZ - left_over_time, 0.0))

viewer.close()