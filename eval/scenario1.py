import mujoco
import mujoco_viewer
import numpy as np

# -----------------------------------------------------
# Load model
# -----------------------------------------------------
model = mujoco.MjModel.from_xml_path("something/xmls/humanoid/humanoid.xml")
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

hz = 50
dt = 1.0 / hz

# -----------------------------------------------------
# Quaternion utilities
# -----------------------------------------------------
def yaw_to_quat(yaw):
    return np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])

def quat_slerp(q1, q2, s):
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        q = q1 + s * (q2 - q1)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * s
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    return q / np.linalg.norm(q)

# -----------------------------------------------------
# Timing parameters
# -----------------------------------------------------
move_duration = 2.0
turn_duration = 0.5
pause_duration = 3.0
speed_scale = 4.0

move_duration /= speed_scale
turn_duration /= speed_scale
pause_duration /= speed_scale

# -----------------------------------------------------
# Define keyframes for translation + orientation
# -----------------------------------------------------
keyframes = []
t = 0.0

# Start facing east
keyframes.append((t, np.array([0.0, 0.0]), 0.0))

# Move east (3s)
t += move_duration
keyframes.append((t, np.array([3.0, 0.0]), 0.0))

# Turn north (fast)
t += turn_duration
keyframes.append((t, np.array([3.0, 0.0]), np.pi / 2))

# Pause while waving (3s)
t += pause_duration
keyframes.append((t, np.array([3.0, 0.0]), np.pi / 2))

# Turn east again
t += turn_duration
keyframes.append((t, np.array([3.0, 0.0]), 0.0))

# Move east (3s)
t += move_duration
keyframes.append((t, np.array([6.0, 0.0]), 0.0))

# Turn north
t += turn_duration
keyframes.append((t, np.array([6.0, 0.0]), np.pi / 2))

# Move north (6s)
t += 2 * move_duration
keyframes.append((t, np.array([6.0, 6.0]), np.pi / 2))

# -----------------------------------------------------
# Waving parameters
# -----------------------------------------------------
wave_start = move_duration + turn_duration       # after turning north
wave_end = wave_start + 2.0 / speed_scale        # 2s of waving
wave_freq = 4.0                                 # waves per second
wave_amp = 0.5                                  # radians amplitude

# Right arm joints (confirmed indices)
shoulder1_idx = 22
shoulder2_idx = 23
shoulder3_idx = 24
elbow_idx = 25

# -----------------------------------------------------
# Interpolation functions
# -----------------------------------------------------
def interpolate_pose(t):
    for i in range(len(keyframes) - 1):
        t0, pos0, yaw0 = keyframes[i]
        t1, pos1, yaw1 = keyframes[i + 1]
        if t0 <= t <= t1:
            s = (t - t0) / (t1 - t0)
            pos = (1 - s) * pos0 + s * pos1
            q0 = yaw_to_quat(yaw0)
            q1 = yaw_to_quat(yaw1)
            quat = quat_slerp(q0, q1, s)
            return pos, quat
    pos, yaw = keyframes[-1][1], keyframes[-1][2]
    return pos, yaw_to_quat(yaw)

def wave_motion(t, qpos):
    """Animate right arm wave during wave window."""
    if wave_start <= t <= wave_end:
        phase = 2 * np.pi * wave_freq * (t - wave_start)

        qpos[shoulder1_idx] = -1.26   # lift arm sideways
        qpos[shoulder2_idx] = -0.157  # slight forward raise
        qpos[shoulder3_idx] = 0.96    # neutral twist 
        elbow_min, elbow_max = -1.2, -0.2
        qpos[elbow_idx] = (elbow_max + elbow_min) / 2 + ((elbow_max - elbow_min) / 2) * np.sin(phase)

    else:
        # Return to neutral pose
        qpos[shoulder1_idx] = 0.0
        qpos[shoulder2_idx] = 0.0
        qpos[shoulder3_idx] = 0.0
        qpos[elbow_idx] = 0.0

    return qpos

# -----------------------------------------------------
# Simulation loop
# -----------------------------------------------------
while viewer.is_alive:
    t = data.time
    pos, quat = interpolate_pose(t)

    # Reset to neutral pose
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    # Base translation and orientation
    data.qpos[0:2] = pos
    data.qpos[2] = 1.5
    data.qpos[3:7] = quat

    # Apply waving if in wave window
    data.qpos = wave_motion(t, data.qpos.copy())

    # Step simulation
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()