#!/usr/bin/env python3
"""
Scenario-based robot motion with LLM or predefined primitives.

Usage:
    python show_scenario.py --generate --context "human waves at 2s"
    python show_scenario.py --predefine --file scenarios/greeting_scenario.txt
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import mujoco
import mujoco_viewer

sys.path.append(str(Path(__file__).parent))

from utils.llm_motion_controller import LLMMotionController
from utils.print_joints import create_name2idx, apply_named_cmd
from primitives.primitive import get_primitive_descriptions, PRIMITIVE_REGISTRY
import primitives.primitive as pm
from eval.motion import Wave, Point

HZ = 50


def create_llm_primitive(instruction, duration, controller):
    """Create a primitive from LLM-generated motion."""
    result = controller.generate_expressive_motion(instruction)

    exec_globals = {"np": np}
    exec(result["generated_code"], exec_globals)
    motion_fn = exec_globals[result["function_name"]]

    class LLMPrimitive:
        def __init__(self):
            self.duration = duration

        def move(self, t):
            qpos = np.zeros(100)
            qpos = motion_fn(t, qpos)
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


def run_scenario(actions, mode, model, name2idx, human_motion=None, loop=True):
    """Run scenario visualization."""
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)

    viewer.cam.distance = 8
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20

    # Create primitives from actions
    llm = LLMMotionController()
    primitives = []

    print("\nGenerating motions...\n")
    for i, action in enumerate(actions):
        print(f"[{i + 1}/{len(actions)}] {action['instruction']}")

        if mode == "predefine":
            # Use predefined primitive
            prim_name = action["instruction"].lower().replace(" ", "_")
            if prim_name in PRIMITIVE_REGISTRY:
                prim = PRIMITIVE_REGISTRY[prim_name](duration=action["duration"])
            else:
                prim = pm.Rest(duration=action["duration"])
        else:
            # Generate with LLM
            prim = create_llm_primitive(action["instruction"], action["duration"], llm)

        primitives.append(
            (action["start_time"], action["start_time"] + action["duration"], prim)
        )

    total_duration = max(end for _, end, _ in primitives)

    human_nq = 35
    robot_start = 35

    print(f"\nRunning scenario ({total_duration:.1f}s, loop={loop})\n")

    while viewer.is_alive:
        t = (data.time % total_duration) if loop else min(data.time, total_duration)

        # Reset
        data.qpos[:] = 0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[robot_start + 3 : robot_start + 7] = [1.0, 0.0, 0.0, 0.0]

        # Human motion
        if human_motion:
            pos, quat = human_motion.interpolate_pose(t)
            human_qpos = np.zeros(human_nq)
            human_qpos[0:3] = pos
            human_qpos[3:7] = quat
            human_qpos = human_motion.motion(t, human_qpos)
            data.qpos[0:human_nq] = human_qpos
        else:
            data.qpos[0:3] = [0.0, 0.0, 1.28]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Robot motion - find active primitive
        robot_cmd = {}
        for start, end, prim in primitives:
            if start <= t < end:
                robot_cmd = prim.move(t - start)
                break

        # Apply robot command
        data.qpos[:] = apply_named_cmd(name2idx, data.qpos, robot_cmd)

        # Set robot base
        data.qpos[robot_start : robot_start + 3] = [3.0, 0.0, 0.793]
        data.qpos[robot_start + 3 : robot_start + 7] = [1.0, 0.0, 0.0, 0.0]

        # Step
        t0 = data.time
        while data.time - t0 < 1 / HZ:
            mujoco.mj_step(model, data)
        viewer.render()

        if not loop and data.time >= total_duration:
            break

    viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Scenario visualization")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--generate", action="store_true", help="Use LLM")
    mode_group.add_argument("--predefine", action="store_true", help="Use primitives")
    parser.add_argument("--context", type=str, help="Scenario description")
    parser.add_argument("--file", type=str, help="Scenario file")
    parser.add_argument("--no-loop", action="store_true", help="Play once")
    args = parser.parse_args()

    # Get context
    if args.file:
        context = Path(args.file).read_text().strip()
    elif args.context:
        context = args.context
    else:
        print("ERROR: Need --context or --file")
        sys.exit(1)

    mode = "generate" if args.generate else "predefine"

    # Setup
    model = mujoco.MjModel.from_xml_path("xmls/scene.xml")
    name2idx = create_name2idx(model)

    # Interpret scenario
    llm = LLMMotionController()
    primitives = get_primitive_descriptions() if mode == "predefine" else None
    actions = llm.interpret_scenario_timeline(context, primitives, mode)

    # Determine human motion
    human_motion = None
    if "wave" in context.lower():
        human_motion = Wave(speed_scale=3.0)
    elif "point" in context.lower():
        human_motion = Point(speed_scale=3.0)

    # Run
    run_scenario(actions, mode, model, name2idx, human_motion, loop=not args.no_loop)


if __name__ == "__main__":
    main()
