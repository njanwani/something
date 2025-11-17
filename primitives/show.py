# from eval.scenario1 import wave_motion, interpolate_pose
import mujoco
import mujoco_viewer
from pathlib import Path
from utils.print_joints import create_name2idx, apply_named_cmd
import primitives.primitive as pm
import time

G1_XYZ_ROOT = "floating_base_joint_xyz"
HZ = 50

path = Path("xmls/scene.xml")
model = mujoco.MjModel.from_xml_path(path.as_posix())
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
viewer.cam = mujoco.MjvCamera()
viewer.cam.distance = 5
viewer.cam.azimuth = 210
viewer.cam.elevation = -45

name2idx = create_name2idx(model)

hz = 50
dt = 1.0 / hz
begin = pm.Rest(duration=1)
wave = pm.Wave(duration=2)
end = pm.Rest(duration=1)

motion = pm.Trajectory(
    begin,
    pm.Transition(begin, wave, duration=0.5),
    wave,
    pm.Transition(wave, end, duration=0.5),
    end,
)
# motion = Point(speed_scale=3.0)
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side_view")

while viewer.is_alive:
    start = time.time()
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qacc[:] = 0

    # CRITICAL: Initialize valid quaternions immediately after reset
    # Humanoid quaternion (indices 3-6)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z - upright
    # Robot quaternion (indices 38-41)
    robot_nq_start = 35  # Start index for robot in combined model
    data.qpos[robot_nq_start + 3 : robot_nq_start + 7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]  # w, x, y, z - upright

    # Human movement (first 35 qpos values) - performing wave animation
    t = data.time
    g1_cmd = motion(t % motion.duration)
    data.qpos[:] = apply_named_cmd(name2idx, data.qpos, g1_cmd)

    # Humanoid base position (should be set after motion for visibility)
    data.qpos[0:3] = [0.0, 0.0, 1.28]  # x, y, z - humanoid height
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Ensure upright orientation

    # Robot base position - positioned in front of humanoid at table
    # G1 robot has 7 qpos for free joint: [x, y, z, qw, qx, qy, qz]
    # Indices 35-41 for G1's base
    data.qpos[robot_nq_start + 0 : robot_nq_start + 3] = [3.0, 0.0, 0.793]  # x, y, z
    data.qpos[robot_nq_start + 3 : robot_nq_start + 7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]  # w, x, y, z - upright

    # Step + render
    t0 = data.time
    while data.time - t0 < 1 / HZ:
        mujoco.mj_step(model, data)
    viewer.render()
    time.sleep(max(1 / HZ - (time.time() - start), 0))

viewer.close()
