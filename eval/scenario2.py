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
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

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

# Pause while pointing and holding (≈3s total)
point_start = t
point_end = t + 3.0 / speed_scale  # total time for point + hold + return
keyframes.append((point_end, np.array([3.0, 0.0]), np.pi / 2))

# Wait 1s before turning east again
t = point_end + 1.0 / speed_scale

# Turn east again
t += turn_duration
keyframes.append((t, np.array([3.0, 0.0]), 0.0))

# Continue moving east and north as before
t += move_duration
keyframes.append((t, np.array([6.0, 0.0]), 0.0))

t += turn_duration
keyframes.append((t, np.array([6.0, 0.0]), np.pi / 2))

t += 2 * move_duration
keyframes.append((t, np.array([6.0, 6.0]), np.pi / 2))

# -----------------------------------------------------
# Arm joint indices
# -----------------------------------------------------
# Right arm
shoulder1_r = 22
shoulder2_r = 23
shoulder3_r = 24
elbow_r = 25

# Left arm
shoulder1_l = 26
shoulder2_l = 27
shoulder3_l = 28
elbow_l = 29

# -----------------------------------------------------
# Arm pose definitions
# -----------------------------------------------------
left_default = dict(
    shoulder1=0.855,
    shoulder2=-0.611,
    shoulder3=-0.244,
    elbow=-1.75
)

right_point = dict(
    shoulder1=0.366,
    shoulder2=0.349,
    shoulder3=-0.0524,
    elbow=-1.75
)

right_default = dict(
    shoulder1=0.75,
    shoulder2=-0.558,
    shoulder3=-0.489,
    elbow=-1.75
)

# -----------------------------------------------------
# Pointing animation parameters
# -----------------------------------------------------
pointing_duration = 0.3 / speed_scale  # fast move to pointing
hold_duration = 2.0 / speed_scale
return_duration = 0.3 / speed_scale

# Timeline
pointing_start = point_start
pointing_hold = pointing_start + pointing_duration
pointing_return = pointing_hold + hold_duration
pointing_end = pointing_return + return_duration

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

def animate_arms(t, qpos):
    """Handles both left default and right arm pointing animation."""
    # --- Left arm always default ---
    qpos[shoulder1_l] = left_default["shoulder1"]
    qpos[shoulder2_l] = left_default["shoulder2"]
    qpos[shoulder3_l] = left_default["shoulder3"]
    qpos[elbow_l] = left_default["elbow"]

    # --- Right arm animation ---
    if pointing_start <= t <= pointing_hold:
        # Move from default to pointing (fast)
        s = (t - pointing_start) / (pointing_hold - pointing_start)
        qpos[shoulder1_r] = (1 - s) * right_default["shoulder1"] + s * right_point["shoulder1"]
        qpos[shoulder2_r] = (1 - s) * right_default["shoulder2"] + s * right_point["shoulder2"]
        qpos[shoulder3_r] = (1 - s) * right_default["shoulder3"] + s * right_point["shoulder3"]
        qpos[elbow_r] = (1 - s) * right_default["elbow"] + s * right_point["elbow"]

    elif pointing_hold < t <= pointing_return:
        # Hold the pointing pose
        qpos[shoulder1_r] = right_point["shoulder1"]
        qpos[shoulder2_r] = right_point["shoulder2"]
        qpos[shoulder3_r] = right_point["shoulder3"]
        qpos[elbow_r] = right_point["elbow"]

    elif pointing_return < t <= pointing_end:
        # Return to default pose (fast)
        s = (t - pointing_return) / (pointing_end - pointing_return)
        qpos[shoulder1_r] = (1 - s) * right_point["shoulder1"] + s * right_default["shoulder1"]
        qpos[shoulder2_r] = (1 - s) * right_point["shoulder2"] + s * right_default["shoulder2"]
        qpos[shoulder3_r] = (1 - s) * right_point["shoulder3"] + s * right_default["shoulder3"]
        qpos[elbow_r] = (1 - s) * right_point["elbow"] + s * right_default["elbow"]

    else:
        # Default pose
        qpos[shoulder1_r] = right_default["shoulder1"]
        qpos[shoulder2_r] = right_default["shoulder2"]
        qpos[shoulder3_r] = right_default["shoulder3"]
        qpos[elbow_r] = right_default["elbow"]

    return qpos

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
    data.qpos = animate_arms(t, data.qpos.copy())

    # Step + render
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()