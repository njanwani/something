"""Integration of generated motions with MuJoCo simulation."""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco_viewer

sys.path.append(str(Path(__file__).parent.parent))

from utils.llm_motion_controller import LLMMotionController


def generate_llm_motions():
    """Generate expressive motions."""
    controller = LLMMotionController()

    motions_to_generate = [
        {
            "instruction": "Notice and acknowledge someone approaching",
            "context": "A person is walking toward you in a friendly manner",
        },
        {
            "instruction": "Show enthusiasm about completing a task",
            "context": "You just successfully finished assembling something",
        },
        {
            "instruction": "Politely indicate you need to leave",
            "context": "You need to excuse yourself from a conversation",
        },
    ]

    results = []
    for motion in motions_to_generate:
        result = controller.generate_expressive_motion(**motion)
        results.append(result)
        print(f"✓ Generated: {result['function_name']}")

    return controller, results


def llm_acknowledge_motion(t, qpos):
    """Acknowledge someone approaching."""
    shoulder1_r = 22
    shoulder2_r = 23
    shoulder3_r = 24
    elbow_r = 25

    shoulder1_l = 26
    shoulder2_l = 27
    shoulder3_l = 28
    elbow_l = 29

    left_default = {
        "shoulder1": 0.855,
        "shoulder2": -0.611,
        "shoulder3": -0.244,
        "elbow": -1.75,
    }
    right_default = {
        "shoulder1": 0.75,
        "shoulder2": -0.558,
        "shoulder3": -0.489,
        "elbow": -1.75,
    }

    wave_pose = {
        "shoulder1": -1.26,
        "shoulder2": -0.157,
        "shoulder3": 0.96,
        "elbow": -0.7,
    }

    qpos[shoulder1_l] = left_default["shoulder1"]
    qpos[shoulder2_l] = left_default["shoulder2"]
    qpos[shoulder3_l] = left_default["shoulder3"]
    qpos[elbow_l] = left_default["elbow"]

    if t < 0.5:
        qpos[shoulder1_r] = right_default["shoulder1"]
        qpos[shoulder2_r] = right_default["shoulder2"]
        qpos[shoulder3_r] = right_default["shoulder3"]
        qpos[elbow_r] = right_default["elbow"]
    elif t < 1.0:
        s = (t - 0.5) / 0.5
        s = 3 * s**2 - 2 * s**3
        qpos[shoulder1_r] = (1 - s) * right_default["shoulder1"] + s * wave_pose[
            "shoulder1"
        ]
        qpos[shoulder2_r] = (1 - s) * right_default["shoulder2"] + s * wave_pose[
            "shoulder2"
        ]
        qpos[shoulder3_r] = (1 - s) * right_default["shoulder3"] + s * wave_pose[
            "shoulder3"
        ]
        qpos[elbow_r] = (1 - s) * right_default["elbow"] + s * wave_pose["elbow"]
    elif t < 2.5:
        phase = 2 * np.pi * 2.0 * (t - 1.0)
        qpos[shoulder1_r] = wave_pose["shoulder1"]
        qpos[shoulder2_r] = wave_pose["shoulder2"]
        qpos[shoulder3_r] = wave_pose["shoulder3"]
        qpos[elbow_r] = -0.7 + 0.3 * np.sin(phase)
    else:
        s = (t - 2.5) / 0.5
        s = min(1.0, 3 * s**2 - 2 * s**3)
        qpos[shoulder1_r] = (1 - s) * wave_pose["shoulder1"] + s * right_default[
            "shoulder1"
        ]
        qpos[shoulder2_r] = (1 - s) * wave_pose["shoulder2"] + s * right_default[
            "shoulder2"
        ]
        qpos[shoulder3_r] = (1 - s) * wave_pose["shoulder3"] + s * right_default[
            "shoulder3"
        ]
        qpos[elbow_r] = (1 - s) * wave_pose["elbow"] + s * right_default["elbow"]

    return qpos


def llm_celebrate_motion(t, qpos):
    """Show enthusiasm about completing task."""
    shoulder1_r = 22
    shoulder2_r = 23
    shoulder3_r = 24
    elbow_r = 25

    shoulder1_l = 26
    shoulder2_l = 27
    shoulder3_l = 28
    elbow_l = 29

    celebrate_pose = {
        "shoulder1": -1.2,
        "shoulder2": 0.0,
        "shoulder3": 1.0,
        "elbow": -0.5,
    }
    default_pose = {
        "shoulder1": 0.75,
        "shoulder2": -0.558,
        "shoulder3": -0.489,
        "elbow": -1.75,
    }

    if t < 0.3:
        s = t / 0.3
        s = 3 * s**2 - 2 * s**3
        qpos[shoulder1_r] = (1 - s) * default_pose["shoulder1"] + s * celebrate_pose[
            "shoulder1"
        ]
        qpos[shoulder2_r] = (1 - s) * default_pose["shoulder2"] + s * celebrate_pose[
            "shoulder2"
        ]
        qpos[shoulder3_r] = (1 - s) * default_pose["shoulder3"] + s * celebrate_pose[
            "shoulder3"
        ]
        qpos[elbow_r] = (1 - s) * default_pose["elbow"] + s * celebrate_pose["elbow"]

        qpos[shoulder1_l] = (1 - s) * default_pose["shoulder1"] + s * celebrate_pose[
            "shoulder1"
        ]
        qpos[shoulder2_l] = (1 - s) * default_pose["shoulder2"] + s * celebrate_pose[
            "shoulder2"
        ]
        qpos[shoulder3_l] = (1 - s) * default_pose["shoulder3"] + s * celebrate_pose[
            "shoulder3"
        ]
        qpos[elbow_l] = (1 - s) * default_pose["elbow"] + s * celebrate_pose["elbow"]
    elif t < 1.5:
        qpos[shoulder1_r] = celebrate_pose["shoulder1"]
        qpos[shoulder2_r] = celebrate_pose["shoulder2"]
        qpos[shoulder3_r] = celebrate_pose["shoulder3"]
        qpos[elbow_r] = celebrate_pose["elbow"]

        qpos[shoulder1_l] = celebrate_pose["shoulder1"]
        qpos[shoulder2_l] = celebrate_pose["shoulder2"]
        qpos[shoulder3_l] = celebrate_pose["shoulder3"]
        qpos[elbow_l] = celebrate_pose["elbow"]
    else:
        s = (t - 1.5) / 0.5
        s = min(1.0, 3 * s**2 - 2 * s**3)
        qpos[shoulder1_r] = (1 - s) * celebrate_pose["shoulder1"] + s * default_pose[
            "shoulder1"
        ]
        qpos[shoulder2_r] = (1 - s) * celebrate_pose["shoulder2"] + s * default_pose[
            "shoulder2"
        ]
        qpos[shoulder3_r] = (1 - s) * celebrate_pose["shoulder3"] + s * default_pose[
            "shoulder3"
        ]
        qpos[elbow_r] = (1 - s) * celebrate_pose["elbow"] + s * default_pose["elbow"]

        qpos[shoulder1_l] = (1 - s) * celebrate_pose["shoulder1"] + s * default_pose[
            "shoulder1"
        ]
        qpos[shoulder2_l] = (1 - s) * celebrate_pose["shoulder2"] + s * default_pose[
            "shoulder2"
        ]
        qpos[shoulder3_l] = (1 - s) * celebrate_pose["shoulder3"] + s * default_pose[
            "shoulder3"
        ]
        qpos[elbow_l] = (1 - s) * celebrate_pose["elbow"] + s * default_pose["elbow"]

    return qpos


def yaw_to_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def dispatch_motion(t, qpos):
    if t < 3.0:
        qpos = llm_acknowledge_motion(t, qpos)
    elif t < 5.0:
        qpos[22:26] = [0.75, -0.558, -0.489, -1.75]
        qpos[26:30] = [0.855, -0.611, -0.244, -1.75]
    elif t < 7.0:
        qpos = llm_celebrate_motion(t - 5.0, qpos)
    else:
        qpos[22:30] = [0.75, -0.558, -0.489, -1.75, 0.855, -0.611, -0.244, -1.75]

    return qpos


def run_simulation():
    """Run simulation."""
    path = Path("xmls/scene.xml")
    model = mujoco.MjModel.from_xml_string(path.read_text())
    data = mujoco.MjData(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)

    while viewer.is_alive:
        t = data.time
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:2] = [0.0, 0.0]
        data.qpos[2] = 1.5
        data.qpos[3:7] = yaw_to_quat(0.0)
        data.qpos = dispatch_motion(t, data.qpos.copy())
        mujoco.mj_step(model, data)
        viewer.render()

    viewer.close()


def main():
    run_simulation()


if __name__ == "__main__":
    main()
