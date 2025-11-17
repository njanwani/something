import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path
from kinematics.chain import Chain
from kinematics.solver import imitate
from kinematics.observe import PoseObserver2D

ARM_POS = 2 * np.random.random(6) - 1
G1_LEFT_ARM_IDX = np.arange(15, 22) + 7

# Load the Unitree G1 model
path = Path("xmls/scene_only_g1.xml")
# model = mujoco.MjModel.from_xml_string(path.read_text())
model = mujoco.MjModel.from_xml_path("xmls/scene.xml")
data = mujoco.MjData(model)

HZ = 50
FLOATING_XYZ = np.array([0.0, 0.0, 0.793])
FLOATING_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
FLOATING_FRAME = np.hstack([FLOATING_XYZ, FLOATING_QUAT])
g1_chain = Chain.from_mujoco(
    base_body="shoulder_start",
    end_body="left_hand",
    model=model,
)

imitate_q = np.zeros(7)  # sol.x
data.qpos[G1_LEFT_ARM_IDX] = imitate_q.copy()

s = np.linspace(0, 1, 7)
sim_start = time.time()
i = 0

po2D = PoseObserver2D(0.4, "accurate")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        po2D.read_pose()
        left_arm_sites = np.concatenate(
            [np.zeros((3, 1)), po2D.left_arm], axis=1
        ) + np.array([0.0, 0.1, 1.1])
        sol = imitate(
            reference_sites=left_arm_sites,
            robot_chain=g1_chain,
            q_init=data.qpos[G1_LEFT_ARM_IDX],
            density=10,
        )
        imitate_q = sol.x
        time_start = data.time
        real_time = time.time()
        while data.time - time_start < 1.0 / HZ:
            mujoco.mj_step(model, data)
            data.qpos[:7] = FLOATING_FRAME
            data.qpos[7:] = 0.0
            data.qvel = 0.0
            data.qacc = 0.0
            data.qpos[G1_LEFT_ARM_IDX] = imitate_q.copy()

        site_names = ["r_arm0", "r_arm1", "r_arm2"]
        for idx, site in enumerate(site_names):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            model.site_pos[site_id] = left_arm_sites[idx]

        mujoco.mj_forward(model, data)

        viewer.sync()
        left_over_time = time.time() - real_time
        time.sleep(max(1.0 / HZ - left_over_time, 0.0))

viewer.close()
