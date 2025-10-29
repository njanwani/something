"""
acknowledge_passerby_motion: Acknowledge the person walking by
"""

import numpy as np



def acknowledge_passerby_motion(t, qpos):
    """Animate robot to acknowledge a passerby with head nod, smile, and wave."""
    # Default positions for left and right arms
    qpos[26:30] = [0.855, -0.611, -0.244, -1.75]  # left arm default
    qpos[22:26] = [0.75, -0.558, -0.489, -1.75]  # right arm default

    # Timing parameters
    face_time = 0.5
    nod_time = 0.5
    smile_time = 0.5
    raise_eyebrow_time = 0.5
    wave_time = 1.0

    # Motion logic
    if t < face_time:
        # Facing the passerby
        qpos[22:26] = [0.75, -0.558, -0.489, -1.75]
    elif t < face_time + nod_time:
        # Acknowledge with head nod
        s = (t - face_time) / nod_time
        s = 3 * s**2 - 2 * s**3  # Smoothstep
        qpos[25] = -1.75 + s * (0.2)  # Subtle nod
    elif t < face_time + nod_time + smile_time:
        # Show slight smile
        pass  # Placeholder for facial expression logic
    elif t < face_time + nod_time + smile_time + raise_eyebrow_time:
        # Raise right eyebrow
        pass  # Placeholder for eyebrow movement logic
    elif t < face_time + nod_time + smile_time + raise_eyebrow_time + wave_time:
        # Wave hand
        wave_start = face_time + nod_time + smile_time + raise_eyebrow_time
        wave_duration = wave_time
        phase = 2 * np.pi * 2.0 * (t - wave_start) / wave_duration
        qpos[22:26] = [-1.26, -0.157, 0.96, -0.7 + 0.3 * np.sin(phase)]

    return qpos


PARAMETERS = {"t": "time in seconds", "qpos": "numpy array of joint positions"}
TIMING = {"start_time": 0, "duration": 2.0}
