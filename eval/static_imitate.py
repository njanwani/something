import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from kinematics.chain import Chain
from kinematics.solver import imitate

ARM_POS = 0.7 * (2 * np.random.random(6) - 1)

# Load the Unitree G1 model
path = Path('xmls/scene.xml')
model = mujoco.MjModel.from_xml_string(path.read_text())
data = mujoco.MjData(model)

def make_chains():
    generic = Chain.from_mujoco(
        base_body = 'base',
        end_body  = 'end_effector',
        model=model,
    )

    g1_left = Chain.from_mujoco(
        base_body = 'left_shoulder_pitch_link',
        end_body  = 'left_hand',
        model=model,
    )
    
    return generic, g1_left

def get_g1_q(d):
    """ IN PROG """
    return d.qpos[23:26]

HZ = 50
FLOATING_XYZ = np.array([0.0, 0.0, 0.8])
FLOATING_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
FLOATING_FRAME = np.hstack([FLOATING_XYZ, FLOATING_QUAT])
PERIOD = 1.5
generic_chain, g1_chain = make_chains()


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