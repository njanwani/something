#!/usr/bin/env python3
"""
Scenario-based LLM motion visualization.

Usage:
    python show_scenario.py --generate --context "a human walks into the room at 0s, stops at a table with the robot facing him at 2s, waves at 3s, then marches along at 5s"
    python show_scenario.py --predefine --context "person approaches and waves, then walks away"
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import mujoco
import mujoco_viewer
import time

sys.path.append(str(Path(__file__).parent))

from utils.llm_motion_controller import LLMMotionController
from utils.print_joints import create_name2idx
from primitives.primitive import (
    PRIMITIVE_REGISTRY,
    get_primitive_descriptions,
    create_motion_function,
)
from eval.motion import Wave, Point

G1_XYZ_ROOT = "floating_base_joint_xyz"
HUMAN_XYZ_ROOT = "floating_base_joint_xyz"
HZ = 50


class ScenarioMotionController:
    """Controls robot motions based on scenario context."""

    def __init__(self, mode="generate", name2idx=None):
        self.mode = mode
        self.llm_controller = (
            LLMMotionController()
        )  # Always need LLM for scenario interpretation
        self.timeline = []  # List of (start_time, end_time, motion_fn)
        self.default_motion = self._create_default_motion()
        self.name2idx = name2idx
        self.available_primitives = get_primitive_descriptions()
        self.human_motion = None  # Will be set based on scenario

    def _create_default_motion(self):
        """Create a simple default motion (idle pose)."""

        def default_motion(t, qpos):
            # Set default arm positions
            qpos[22:26] = [0.75, -0.558, -0.489, -1.75]  # right arm
            qpos[26:30] = [0.855, -0.611, -0.244, -1.75]  # left arm
            return qpos

        return default_motion

    def interpret_scenario(self, context: str):
        """
        Use LLM to interpret the scenario and determine robot actions with timing.

        Args:
            context: Natural language description of the scenario with timing
                    e.g., "a human walks into the room at 0s, stops at 2s, waves at 3s"

        Returns:
            List of robot actions with timing
        """
        # Delegate to LLMMotionController for scenario interpretation
        return self.llm_controller.interpret_scenario_timeline(
            context=context,
            available_primitives=self.available_primitives
            if self.mode == "predefine"
            else None,
            mode=self.mode,
        )

    def generate_motions_from_timeline(self, actions):
        """Generate motion functions for each action in the timeline."""
        print("Generating robot motions...\n")

        for i, action in enumerate(actions):
            instruction = action["instruction"]
            start_time = action["start_time"]
            duration = action["duration"]

            print(f"[{i + 1}/{len(actions)}] Generating: {instruction}")

            # Check if using predefined primitives
            if self.mode == "predefine":
                # Try to match instruction to a primitive
                instruction_lower = instruction.lower().replace(" ", "_")
                matched_primitive = None

                # Direct match
                if instruction_lower in PRIMITIVE_REGISTRY:
                    matched_primitive = instruction_lower
                else:
                    # Fuzzy match
                    for prim_name in PRIMITIVE_REGISTRY.keys():
                        if (
                            prim_name in instruction_lower
                            or instruction_lower in prim_name
                        ):
                            matched_primitive = prim_name
                            break

                if matched_primitive:
                    motion_fn = create_motion_function(
                        matched_primitive, duration, self.name2idx
                    )
                    print(f"   [OK] Using primitive: {matched_primitive}")
                else:
                    # Fall back to default
                    motion_fn = self.default_motion
                    print(
                        "   [WARNING] No matching primitive found, using default idle"
                    )

            # Check if it's an idle/default action
            elif any(
                word in instruction.lower()
                for word in ["idle", "wait", "rest", "default", "stand"]
            ):
                motion_fn = self.default_motion
                print("   [OK] Using default idle pose")

            else:
                # Generate motion with LLM
                result = self.llm_controller.generate_expressive_motion(
                    instruction=instruction, context=action.get("reasoning", "")
                )

                # Execute the generated code
                motion_code = result["generated_code"]
                exec_globals = {"np": np}
                exec(motion_code, exec_globals)

                function_name = result["function_name"]
                motion_fn = exec_globals[function_name]
                print(f"   [OK] Generated {function_name}")

            self.timeline.append(
                {
                    "start_time": start_time,
                    "end_time": start_time + duration,
                    "motion_fn": motion_fn,
                    "instruction": instruction,
                }
            )

        print(f"\nGenerated {len(self.timeline)} motions successfully!\n")

    def get_motion_at_time(self, t: float):
        """Get the appropriate motion function for time t."""
        for item in self.timeline:
            if item["start_time"] <= t < item["end_time"]:
                local_t = t - item["start_time"]
                return item["motion_fn"], local_t

        # Default to idle if outside timeline
        return self.default_motion, t

    def get_total_duration(self):
        """Get total duration of the scenario."""
        if not self.timeline:
            return 10.0  # default
        return max(item["end_time"] for item in self.timeline)

    def set_human_motion_from_context(self, context: str):
        """Determine human motion type from scenario context."""
        context_lower = context.lower()

        # Check for specific actions mentioned
        if "wave" in context_lower or "waves" in context_lower:
            print("\nUsing Wave motion for human")
            self.human_motion = Wave(speed_scale=3.0)
        elif "point" in context_lower:
            print("\nUsing Point motion for human")
            self.human_motion = Point(speed_scale=3.0)
        else:
            # Default to wave if no specific action detected
            print("\nUsing Wave motion for human (default)")
            self.human_motion = Wave(speed_scale=3.0)


def yaw_to_quat(yaw):
    """Convert yaw angle to quaternion."""
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def run_scenario_visualization(controller: ScenarioMotionController, model, loop=True):
    """Run the MuJoCo visualization with scenario-based motions."""
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)

    # Camera settings
    viewer.cam = mujoco.MjvCamera()
    viewer.cam.distance = 8
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20

    name2idx = controller.name2idx
    total_duration = controller.get_total_duration()

    # Find qpos offset for robot (assumes humanoid is loaded first, then G1)
    # Humanoid has 35 qpos values (7 for free joint + 28 for joints)
    human_nq = 35  # This may need to be determined dynamically
    robot_nq_start = human_nq

    print(f"Starting visualization (scenario duration: {total_duration:.1f}s)")
    print(f"Model has {model.nq} qpos values, {model.nbody} bodies")
    if loop:
        print("   Looping enabled - scenario will repeat")
    print("   Close window to exit\n")

    try:
        while viewer.is_alive:
            start = time.time()
            data.qpos[:] = 0
            data.qvel[:] = 0
            data.qacc[:] = 0

            # CRITICAL: Initialize valid quaternions immediately after reset
            # Invalid quaternions [0,0,0,0] cause undefined orientation (often upside down)
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Humanoid quaternion - upright
            data.qpos[robot_nq_start + 3 : robot_nq_start + 7] = [
                1.0,
                0.0,
                0.0,
                0.0,
            ]  # Robot quaternion - upright

            # Get current time in scenario
            if loop:
                t = data.time % total_duration if total_duration > 0 else data.time
            else:
                t = min(data.time, total_duration)

            # === HUMAN MOTION (first body in scene) ===
            if controller.human_motion:
                # Create temporary qpos for human with valid quaternion
                human_qpos = np.zeros(human_nq)
                human_qpos[3:7] = [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]  # Initialize with valid quaternion

                # Get human position and orientation
                pos, quat = controller.human_motion.interpolate_pose(t)
                human_qpos[0:3] = pos
                human_qpos[3:7] = quat

                # Apply human arm motion
                human_qpos = controller.human_motion.motion(t, human_qpos.copy())

                # Copy to main data
                data.qpos[0:human_nq] = human_qpos
            else:
                # Default human position
                data.qpos[0:3] = [0.0, 0.0, 1.28]  # x, y, z
                data.qpos[3:7] = yaw_to_quat(0.0)  # Valid quaternion

            # === ROBOT MOTION (second body in scene, G1) ===
            # Get appropriate robot motion
            motion_fn, local_t = controller.get_motion_at_time(t)

            # Work with full qpos array with valid quaternions initialized
            temp_qpos = np.zeros(model.nq, dtype=np.float64)
            # Initialize both quaternions in temp array
            temp_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Humanoid
            temp_qpos[robot_nq_start + 3 : robot_nq_start + 7] = [
                1.0,
                0.0,
                0.0,
                0.0,
            ]  # Robot

            # Apply robot motion (modifies robot joint angles, not base)
            try:
                temp_qpos = motion_fn(local_t, temp_qpos.copy())
            except Exception as e:
                print(f"\nERROR in robot motion function: {e}")
                # Use default motion as fallback
                temp_qpos = controller.default_motion(local_t, temp_qpos.copy())

            # Set robot base position and orientation (after motion so it doesn't get overwritten)
            temp_qpos[robot_nq_start + 0 : robot_nq_start + 3] = [
                3.0,
                0.0,
                0.793,
            ]  # x, y, z position
            temp_qpos[robot_nq_start + 3 : robot_nq_start + 7] = [
                1.0,
                0.0,
                0.0,
                0.0,
            ]  # w, x, y, z - upright orientation

            # Copy robot portion to main data
            data.qpos[robot_nq_start:] = temp_qpos[robot_nq_start:]

            # Step + render
            t0 = data.time
            while data.time - t0 < 1 / HZ:
                mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(max(1 / HZ - (time.time() - start), 0))

            # Stop if not looping and past duration
            if not loop and data.time >= total_duration:
                print(f"\nScenario completed ({total_duration:.1f}s)")
                time.sleep(2)  # Show final pose briefly
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        viewer.close()
        print("Visualization closed")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scenario-based robot motion visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate motions from context string
  python show_scenario.py --generate --context "a human walks into the room at 0s, stops at 2s, waves at 3s"
  
  # Load scenario from file
  python show_scenario.py --generate --file scenarios/greeting_scenario.txt
  
  # Don't loop
  python show_scenario.py --generate --file scenarios/pointing_scenario.txt --no-loop
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--generate", action="store_true", help="Generate motions using LLM"
    )
    mode_group.add_argument(
        "--predefine", action="store_true", help="Use predefined motion library"
    )

    parser.add_argument(
        "--context",
        type=str,
        help='Scenario description with timing (e.g., "human enters at 0s, waves at 2s")',
    )

    parser.add_argument("--file", type=str, help="Path to scenario file (.txt)")

    parser.add_argument(
        "--no-loop", action="store_true", help="Play scenario once without looping"
    )

    parser.add_argument("--save", type=str, help="Save generated scenario to file")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Get context from file or argument
    if args.file:
        with open(args.file, "r") as f:
            context = f.read().strip()
        print(f"Loaded scenario from: {args.file}")
    elif args.context:
        context = args.context
    else:
        print("ERROR: Must provide either --context or --file")
        sys.exit(1)

    # Determine mode
    mode = "generate" if args.generate else "predefine"

    print("\n" + "=" * 70)
    print("SCENARIO-BASED ROBOT MOTION VISUALIZATION")
    print("=" * 70)
    print(f"\nMode: {mode.upper()}")
    print(f"Context: {context}")
    print(f"Loop: {'No' if args.no_loop else 'Yes'}")
    print("=" * 70)

    try:
        # Load model early to get name2idx for primitives
        path = Path("xmls/scene.xml")
        model = mujoco.MjModel.from_xml_path(path.as_posix())
        name2idx = create_name2idx(model)

        # Create controller
        controller = ScenarioMotionController(mode=mode, name2idx=name2idx)

        # Interpret scenario
        actions = controller.interpret_scenario(context)

        # Generate motions
        controller.generate_motions_from_timeline(actions)

        # Determine human motion based on context
        controller.set_human_motion_from_context(context)

        # Optionally save
        if args.save:
            print(f"\nSaving scenario to {args.save}...")
            # TODO: Implement save functionality

        # Run visualization
        run_scenario_visualization(controller, model, loop=not args.no_loop)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
