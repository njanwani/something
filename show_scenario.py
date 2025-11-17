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
import time

sys.path.append(str(Path(__file__).parent))

from utils.llm_motion_controller import LLMMotionController
from utils.print_joints import create_name2idx
from primitives.primitive import get_primitive_descriptions, create_motion_function
from eval.motion import Wave, Point

HZ = 50


class ScenarioController:
    """Interprets scenarios and generates robot motions."""
    
    def __init__(self, mode="generate", name2idx=None):
        self.mode = mode
        self.llm = LLMMotionController()
        self.name2idx = name2idx
        self.timeline = []
        self.human_motion = None
    
    def interpret_scenario(self, context):
        """Get timeline of robot actions from scenario description."""
        primitives = get_primitive_descriptions() if self.mode == "predefine" else None
        return self.llm.interpret_scenario_timeline(context, primitives, self.mode)
    
    def generate_motions(self, actions):
        """Generate motion functions from action timeline."""
        print("Generating motions...\n")
        
        for i, action in enumerate(actions):
            print(f"[{i + 1}/{len(actions)}] {action['instruction']}")
            
            # Predefine mode: use primitives
            if self.mode == "predefine":
                motion_fn = create_motion_function(
                    action['instruction'].lower().replace(' ', '_'),
                    action['duration'],
                    self.name2idx
                )
            
            # Generate mode: use LLM
            else:
                result = self.llm.generate_expressive_motion(action['instruction'])
                exec_globals = {"np": np}
                exec(result["generated_code"], exec_globals)
                motion_fn = exec_globals[result["function_name"]]
            
            self.timeline.append({
                "start": action["start_time"],
                "end": action["start_time"] + action["duration"],
                "motion": motion_fn
            })
        
        print(f"\nGenerated {len(self.timeline)} motions!\n")
    
    def get_motion_at(self, t):
        """Get motion function for time t."""
        for item in self.timeline:
            if item["start"] <= t < item["end"]:
                return item["motion"], t - item["start"]
        return lambda t, q: q, t  # Default: do nothing
    
    def set_human_motion(self, context):
        """Determine human motion from scenario."""
        if "wave" in context.lower():
            self.human_motion = Wave(speed_scale=3.0)
        elif "point" in context.lower():
            self.human_motion = Point(speed_scale=3.0)


def run_scenario(controller, model, loop=True):
    """Run scenario visualization."""
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    
    viewer.cam.distance = 8
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    
    duration = max(item["end"] for item in controller.timeline)
    human_nq = 35  # Humanoid qpos size
    robot_start = 35  # Robot starts at index 35
    
    print(f"Running scenario ({duration:.1f}s, loop={loop})\n")
    
    while viewer.is_alive:
        t = (data.time % duration) if loop else min(data.time, duration)
        
        # Reset
        data.qpos[:] = 0
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Human upright
        data.qpos[robot_start + 3:robot_start + 7] = [1.0, 0.0, 0.0, 0.0]  # Robot upright
        
        # Human motion
        if controller.human_motion:
            pos, quat = controller.human_motion.interpolate_pose(t)
            human_qpos = np.zeros(human_nq)
            human_qpos[0:3] = pos
            human_qpos[3:7] = quat
            human_qpos = controller.human_motion.motion(t, human_qpos)
            data.qpos[0:human_nq] = human_qpos
        else:
            data.qpos[0:3] = [0.0, 0.0, 1.28]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        # Robot motion
        motion_fn, local_t = controller.get_motion_at(t)
        temp_qpos = np.zeros(model.nq)
        temp_qpos[robot_start + 3:robot_start + 7] = [1.0, 0.0, 0.0, 0.0]
        temp_qpos = motion_fn(local_t, temp_qpos)
        
        # Set robot base
        temp_qpos[robot_start:robot_start + 3] = [3.0, 0.0, 0.793]
        temp_qpos[robot_start + 3:robot_start + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[robot_start:] = temp_qpos[robot_start:]
        
        # Step
        t0 = data.time
        while data.time - t0 < 1 / HZ:
            mujoco.mj_step(model, data)
        viewer.render()
        
        if not loop and data.time >= duration:
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
    controller = ScenarioController(mode, name2idx)
    
    # Generate
    actions = controller.interpret_scenario(context)
    controller.generate_motions(actions)
    controller.set_human_motion(context)
    
    # Run
    run_scenario(controller, model, loop=not args.no_loop)


if __name__ == "__main__":
    main()
