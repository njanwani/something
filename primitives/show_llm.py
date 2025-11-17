"""
LLM-powered motion visualizer for Unitree G1.
Generates and visualizes expressive motions from natural language.
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco_viewer
import time

sys.path.append(str(Path(__file__).parent.parent))

from utils.llm_motion_controller import LLMMotionController
from utils.print_joints import create_name2idx, apply_named_cmd
import primitives.primitive as pm

HZ = 50


def create_llm_primitive(instruction, duration, controller):
    """Create a primitive from LLM-generated motion."""
    result = controller.generate_expressive_motion(instruction)

    # Execute generated code
    exec_globals = {"np": np}
    exec(result["generated_code"], exec_globals)
    motion_fn = exec_globals[result["function_name"]]

    # Wrap in primitive-like class
    class LLMPrimitive:
        def __init__(self):
            self.duration = duration
            self.motion_fn = motion_fn

        def move(self, t):
            # Create temp qpos, apply motion, extract joint commands
            qpos = np.zeros(100)  # Oversized temp array
            qpos = self.motion_fn(t, qpos)

            # Extract arm joints (indices 22-29)
            return {
                "right_shoulder_pitch_joint": qpos[22],
                "right_shoulder_roll_joint": qpos[23],
                "right_shoulder_yaw_joint": qpos[24],
                "right_elbow_joint": qpos[25],
                "left_shoulder_pitch_joint": qpos[26],
                "left_shoulder_roll_joint": qpos[27],
                "left_shoulder_yaw_joint": qpos[28],
                "left_elbow_joint": qpos[29],
            }

    return LLMPrimitive()


def main():
    """Example usage."""
    controller = LLMMotionController()

    # Generate motions as primitives
    print("\nGenerating motions...\n")

    print("[1/3] Wave hello")
    wave = create_llm_primitive("Wave hello to someone", 2.0, controller)

    print("[2/3] Point")
    point = create_llm_primitive("Point at something on a table", 2.0, controller)

    print("[3/3] Celebrate")
    celebrate = create_llm_primitive("Show excitement", 2.0, controller)

    # Create trajectory like show.py
    rest = pm.Rest(duration=1.0)

    motion = pm.Trajectory(
        rest,
        pm.Transition(rest, wave, duration=0.5),
        wave,
        pm.Transition(wave, point, duration=0.5),
        point,
        pm.Transition(point, celebrate, duration=0.5),
        celebrate,
        pm.Transition(celebrate, rest, duration=0.5),
        rest,
    )

    print(f"\nCreated trajectory (duration: {motion.duration}s)\n")

    # Load and run
    path = Path("xmls/g1_standalone.xml")
    model = mujoco.MjModel.from_xml_path(path.as_posix())
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)

    viewer.cam.distance = 3
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20

    name2idx = create_name2idx(model)

    while viewer.is_alive:
        start = time.time()

        # Reset
        data.qpos[:] = 0
        data.qpos[0:3] = [0.0, 0.0, 0.793]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Get command from trajectory
        t = data.time % motion.duration
        cmd = motion(t)

        # Apply command (like show.py does)
        data.qpos[:] = apply_named_cmd(name2idx, data.qpos, cmd)

        # Keep base fixed
        data.qpos[0:3] = [0.0, 0.0, 0.793]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Step
        t0 = data.time
        while data.time - t0 < 1 / HZ:
            mujoco.mj_step(model, data)

        viewer.render()
        time.sleep(max(0, 1 / HZ - (time.time() - start)))

    viewer.close()


if __name__ == "__main__":
    main()
