import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from kinematics.chain import Chain
from kinematics.solver import imitate

ARM_POS = 0.7 * (2 * np.random.random(6) - 1)
G1_LEFT_ARM_IDX = np.arange(18, 21) + 7
GENERIC_ARM_IDX = np.arange(21, 27) + 7

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
        base_body = 'left_upper_arm',
        end_body  = 'left_hand',
        model=model,
    )
    
    return generic, g1_left

HZ = 50
FLOATING_XYZ = np.array([0.0, 0.0, 1.5])
FLOATING_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
FLOATING_FRAME = np.hstack([FLOATING_XYZ, FLOATING_QUAT])
generic_chain, g1_chain = make_chains()
# imitate_q = np.zeros(6)
# imitate_q[0] = np.pi / 2
# imitate_q[3] = np.pi / 2
# imitate_q[4] = -np.pi / 2
# imitate_q[5] = np.pi / 2
# quit()
# # quit()
imitate_q = np.zeros(3) #np.random.random(3) * 2 - 1
imitate_q[0] = np.pi / 2
data.qpos[G1_LEFT_ARM_IDX] = imitate_q.copy()

# data.qpos[GENERIC_ARM_IDX] = imitate_q.copy()
s = np.linspace(0, 1, 50)
generic_path = generic_chain.compute_path(data.qpos[GENERIC_ARM_IDX], s)
g1_path      = g1_chain.compute_path(data.qpos[G1_LEFT_ARM_IDX], s)
# print(np.round(generic_path, 2))
# print('-----')
# print(np.round(g1_path, 2))
# site_pos = generic_path[-1, :3, 3]
site_pos = g1_path[-1, :3, 3]
# site_pos = g1_chain.base_T[:3, 3]
# site_pos = generic_chain.base_T[:3, 3]

sim_start = time.time()
# Simple walking-ish gait: sinusoidal hip/knee swings
i = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        i += 1
        if i >= len(generic_path):
            i = 0
        time_start = data.time
        real_time  = time.time()
        while data.time - time_start < 1.0 / HZ:
            mujoco.mj_step(model, data)
            data.qpos[:7] = FLOATING_FRAME
            data.qpos[7:] = 0.0
            data.qvel = 0.0
            data.qacc = 0.0
            # data.qpos[GENERIC_ARM_IDX] = imitate_q.copy()
            data.qpos[G1_LEFT_ARM_IDX] = imitate_q.copy()
        model.site_pos[0] = g1_path[i, :3, 3].copy()
        # model.site_pos[0] = g1_chain.base_T[:3, 3]
        mujoco.mj_forward(model, data)

        viewer.sync()
        left_over_time = time.time() - real_time
        time.sleep(max(1.0 / HZ - left_over_time, 0.0))

viewer.close()