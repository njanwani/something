import mujoco
import mujoco_viewer
import numpy as np

# # -----------------------------------------------------
# # Load model
# # -----------------------------------------------------
# model = mujoco.MjModel.from_xml_path("xmls/humanoid/humanoid.xml")
# data = mujoco.MjData(model)
# viewer = mujoco_viewer.MujocoViewer(model, data)

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
# Waving parameters (fixed to start *after* turn completes)
# -----------------------------------------------------
turn_buffer = 0.2 / speed_scale           # short delay after turn before waving
pre_wave_duration = 0.3 / speed_scale     # smooth lift before waving
post_wave_duration = 0.3 / speed_scale    # smooth return after waving
wave_duration = 2.0 / speed_scale         # duration of actual waving

wave_freq = 4.0                           # waves per second
wave_amp = 0.5                            # radians amplitude

# Start waving *after* turning north and buffer delay
wave_start = move_duration + turn_duration + turn_buffer
wave_end = wave_start + wave_duration

# -----------------------------------------------------
# Arm joint indices
# -----------------------------------------------------
# Right arm joints
shoulder1_r = 22
shoulder2_r = 23
shoulder3_r = 24
elbow_r = 25

# Left arm joints
shoulder1_l = 26
shoulder2_l = 27
shoulder3_l = 28
elbow_l = 29

# -----------------------------------------------------
# Default arm poses (from pointing code)
# -----------------------------------------------------
left_default = dict(
    shoulder1=0.855,
    shoulder2=-0.611,
    shoulder3=-0.244,
    elbow=-1.75
)

right_default = dict(
    shoulder1=0.75,
    shoulder2=-0.558,
    shoulder3=-0.489,
    elbow=-1.75
)

# Wave lifted pose
wave_pose = dict(
    shoulder1=-1.26,
    shoulder2=-0.157,
    shoulder3=0.96,
    elbow=-0.7
)

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

def smooth_lerp(v0, v1, s):
    """Smooth interpolation between two values."""
    s = np.clip(s, 0.0, 1.0)
    s = 3 * s**2 - 2 * s**3  # smoothstep
    return (1 - s) * v0 + s * v1

def wave_motion(t, qpos):
    """Animate right arm wave during wave window with smooth transitions."""
    # Left arm always at default
    qpos[shoulder1_l] = left_default["shoulder1"]
    qpos[shoulder2_l] = left_default["shoulder2"]
    qpos[shoulder3_l] = left_default["shoulder3"]
    qpos[elbow_l] = left_default["elbow"]

    # ---- Pre-wave transition ----
    if wave_start - pre_wave_duration <= t < wave_start:
        s = (t - (wave_start - pre_wave_duration)) / pre_wave_duration
        for k, v in wave_pose.items():
            qpos[globals()[k + "_r"]] = smooth_lerp(right_default[k], v, s)

    # ---- Waving ----
    elif wave_start <= t <= wave_end:
        phase = 2 * np.pi * wave_freq * (t - wave_start)
        qpos[shoulder1_r] = wave_pose["shoulder1"]
        qpos[shoulder2_r] = wave_pose["shoulder2"]
        qpos[shoulder3_r] = wave_pose["shoulder3"]
        elbow_min, elbow_max = -1.4, -0.2
        qpos[elbow_r] = (elbow_max + elbow_min) / 2 + ((elbow_max - elbow_min) / 2) * np.sin(phase)

    # ---- Post-wave return ----
    elif wave_end < t <= wave_end + post_wave_duration:
        s = (t - wave_end) / post_wave_duration
        for k, v in wave_pose.items():
            qpos[globals()[k + "_r"]] = smooth_lerp(v, right_default[k], s)

    # ---- Default (not waving) ----
    else:
        qpos[shoulder1_r] = right_default["shoulder1"]
        qpos[shoulder2_r] = right_default["shoulder2"]
        qpos[shoulder3_r] = right_default["shoulder3"]
        qpos[elbow_r] = right_default["elbow"]

    return qpos

# # -----------------------------------------------------
# # Simulation loop
# # -----------------------------------------------------
# while viewer.is_alive:
#     t = data.time
#     pos, quat = interpolate_pose(t)

#     # Reset to neutral pose
#     data.qpos[:] = 0.0
#     data.qvel[:] = 0.0

#     # Base translation and orientation
#     data.qpos[0:2] = pos
#     data.qpos[2] = 1.5
#     data.qpos[3:7] = quat

#     # Apply waving if in wave window
#     data.qpos = wave_motion(t, data.qpos.copy())

#     # Step simulation
#     mujoco.mj_step(model, data)
#     viewer.render()

# viewer.close()