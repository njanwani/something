"""
Interactive LLM-powered motion visualizer.
Shows LLM-generated expressive motions in real-time.
"""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco_viewer
import time

sys.path.append(str(Path(__file__).parent.parent))

from utils.llm_motion_controller import LLMMotionController
from utils.print_joints import create_name2idx

G1_XYZ_ROOT = "floating_base_joint_xyz"
HZ = 50


class LLMMotionPlayer:
    """Plays LLM-generated motion functions."""

    def __init__(self):
        self.motions = {}
        self.motion_order = []
        self.controller = None

    def add_motion(self, name: str, motion_fn, duration: float):
        """Add a motion to the player."""
        self.motions[name] = {"function": motion_fn, "duration": duration}
        self.motion_order.append(name)

    def get_motion_at_time(self, t: float):
        """Get the motion function and local time for global time t."""
        cumulative_time = 0

        for name in self.motion_order:
            motion = self.motions[name]
            if t < cumulative_time + motion["duration"]:
                local_t = t - cumulative_time
                return motion["function"], local_t
            cumulative_time += motion["duration"]

        # Loop back to start
        total_duration = sum(m["duration"] for m in self.motions.values())
        if total_duration > 0:
            t_mod = t % total_duration
            return self.get_motion_at_time(t_mod)

        # Fallback to first motion
        first_motion = self.motions[self.motion_order[0]]
        return first_motion["function"], 0.0

    def get_total_duration(self):
        """Get total duration of all motions."""
        return sum(m["duration"] for m in self.motions.values())


def yaw_to_quat(yaw):
    """Convert yaw angle to quaternion."""
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def generate_motions_from_prompts(prompts):
    """
    Generate motions from natural language prompts.

    Args:
        prompts: List of (instruction, context, duration) tuples

    Returns:
        LLMMotionPlayer with generated motions
    """
    controller = LLMMotionController()
    player = LLMMotionPlayer()
    player.controller = controller

    print("\nGenerating motions with LLM...\n")

    for i, (instruction, context, duration) in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] Generating: {instruction}")

        result = controller.generate_expressive_motion(
            instruction=instruction, context=context
        )

        # Create motion function from generated code
        motion_code = result["generated_code"]
        exec_globals = {"np": np}
        exec(motion_code, exec_globals)

        function_name = result["function_name"]
        motion_fn = exec_globals[function_name]

        player.add_motion(instruction, motion_fn, duration)
        print(f"   [OK] {function_name} (duration: {duration}s)")

    print(f"\nGenerated {len(prompts)} motions successfully!\n")
    return player


def load_motion_from_file(filepath: str):
    """Load a pre-generated motion from file."""
    with open(filepath, "r") as f:
        code = f.read()

    exec_globals = {"np": np}
    exec(code, exec_globals)

    # Find the motion function (look for functions that take t and qpos)
    for name, obj in exec_globals.items():
        if callable(obj) and name.endswith("_motion"):
            return obj

    raise ValueError(f"No motion function found in {filepath}")


def run_visualization(player: LLMMotionPlayer, camera_settings=None):
    """Run the MuJoCo visualization with LLM motions."""
    path = Path("xmls/scene.xml")
    model = mujoco.MjModel.from_xml_path(path.as_posix())
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)

    # Camera settings
    viewer.cam = mujoco.MjvCamera()
    if camera_settings:
        viewer.cam.distance = camera_settings.get("distance", 5)
        viewer.cam.azimuth = camera_settings.get("azimuth", 210)
        viewer.cam.elevation = camera_settings.get("elevation", -45)
    else:
        viewer.cam.distance = 5
        viewer.cam.azimuth = 210
        viewer.cam.elevation = -45

    name2idx = create_name2idx(model)
    total_duration = player.get_total_duration()

    print(f"Starting visualization (total duration: {total_duration}s, looping)")
    print("   Close window to exit\n")

    while viewer.is_alive:
        start = time.time()
        data.qpos[:] = 0
        data.qvel[:] = 0
        data.qacc[:] = 0

        # Get current motion
        t = data.time % total_duration if total_duration > 0 else data.time
        motion_fn, local_t = player.get_motion_at_time(t)

        # Apply LLM-generated motion
        data.qpos[:] = 0.0
        data.qpos[0:2] = [0.0, 0.0]  # x, y position
        data.qpos[2] = 1.5  # z position (height)
        data.qpos[3:7] = yaw_to_quat(0.0)  # orientation
        data.qpos = motion_fn(local_t, data.qpos.copy())

        # Robot base height (if using G1 robot)
        try:
            data.qpos[name2idx[G1_XYZ_ROOT][2]] = 0.793
        except KeyError:
            pass  # Not using G1 robot

        # Step + render
        t0 = data.time
        while data.time - t0 < 1 / HZ:
            mujoco.mj_step(model, data)
        viewer.render()
        time.sleep(max(1 / HZ - (time.time() - start), 0))

    viewer.close()
    print("\nVisualization closed")


def main():
    """Main function with example usage."""

    # Example 1: Generate motions from prompts
    prompts = [
        (
            "Wave hello to someone approaching",
            "A person is walking toward you from across the room",
            3.0,  # duration in seconds
        ),
        (
            "Point at an object on a table",
            "You need to direct someone's attention to an object",
            2.5,
        ),
        (
            "Show excitement about finishing a task",
            "You just completed building something successfully",
            2.0,
        ),
    ]

    player = generate_motions_from_prompts(prompts)

    # Example 2: Or load from pre-generated file
    # player = LLMMotionPlayer()
    # motion_fn = load_motion_from_file('generated_motion.py')
    # player.add_motion("acknowledge", motion_fn, 3.0)

    # Run visualization
    camera_settings = {"distance": 5, "azimuth": 210, "elevation": -45}

    run_visualization(player, camera_settings)


if __name__ == "__main__":
    main()
