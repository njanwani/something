from eval.scenario1 import wave_motion, interpolate_pose
from eval.scenario2 import point_motion
import mujoco
import mujoco_viewer
from pathlib import Path


# -----------------------------------------------------
# Load model
# -----------------------------------------------------
# model = mujoco.MjModel.from_xml_string("xmls/scene.xml")
path = Path("xmls/scene.xml")
model = mujoco.MjModel.from_xml_string(path.read_text())
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

hz = 50
dt = 1.0 / hz
# -----------------------------------------------------
name2motion = {"wave": wave_motion, "point": point_motion}
MOTION = "point"  # Choose motion type: "wave" or "point"
motion_func = name2motion[MOTION]
# -----------------------------------------------------
# Simulation loop
# -----------------------------------------------------
while viewer.is_alive:
    t = data.time
    pos, quat = interpolate_pose(t)

    # Reset base
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    # Base translation + orientation
    data.qpos[0:2] = pos
    data.qpos[2] = 1.5
    data.qpos[3:7] = quat

    # Arm animation
    data.qpos = motion_func(t, data.qpos.copy())

    # Step + render
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
