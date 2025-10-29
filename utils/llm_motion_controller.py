"""Motion controller for expressive humanoid robot behaviors."""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMMotionController:
    """Generates expressive motion code for humanoid robots."""

    MODELS = {
        "mini": "gpt-4o-mini",
        "turbo": "gpt-3.5-turbo",
        "gpt4o": "gpt-4o",
        "gpt4": "gpt-4-turbo",
    }

    JOINT_INDICES = {
        # Right arm
        "right_shoulder1": 22,
        "right_shoulder2": 23,
        "right_shoulder3": 24,
        "right_elbow": 25,
        # Left arm
        "left_shoulder1": 26,
        "left_shoulder2": 27,
        "left_shoulder3": 28,
        "left_elbow": 29,
    }

    DEFAULT_POSES = {
        "left_arm_default": {
            "shoulder1": 0.855,
            "shoulder2": -0.611,
            "shoulder3": -0.244,
            "elbow": -1.75,
        },
        "right_arm_default": {
            "shoulder1": 0.75,
            "shoulder2": -0.558,
            "shoulder3": -0.489,
            "elbow": -1.75,
        },
        "right_arm_wave": {
            "shoulder1": -1.26,
            "shoulder2": -0.157,
            "shoulder3": 0.96,
            "elbow": -0.7,
        },
        "right_arm_point": {
            "shoulder1": 0.366,
            "shoulder2": 0.349,
            "shoulder3": -0.0524,
            "elbow": -1.75,
        },
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "mini"):
        """Initialize the motion controller.

        Args:
            api_key: OpenAI API key (or set OPENAI_KEY in .env)
            model: Model to use. Options:
                - 'mini' (gpt-4o-mini, default, cheapest)
                - 'turbo' (gpt-3.5-turbo)
                - 'gpt4o' (gpt-4o, most capable)
                - 'gpt4' (gpt-4-turbo)
                Or pass exact model name string
        """
        self.api_key = api_key or os.environ.get("OPENAI_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in .env file as OPENAI_KEY"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = self.MODELS.get(model, model)
        self.skill_library: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, str]] = []

    def generate_expressive_motion(
        self,
        instruction: str,
        context: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate expressive motion from natural language instruction."""
        social_response = self._step1_social_reasoning(instruction, context)
        robot_procedure = self._step2_robot_translation(
            instruction, social_response, constraints
        )
        code_result = self._step3_code_generation(
            instruction, robot_procedure, social_response
        )

        result = {
            "instruction": instruction,
            "social_reasoning": social_response["reasoning"],
            "human_behavior": social_response["behavior"],
            "robot_procedure": robot_procedure,
            "generated_code": code_result["code"],
            "function_name": code_result["function_name"],
            "parameters": code_result.get("parameters", {}),
            "timing": code_result.get("timing", {}),
        }

        self._add_to_skill_library(result)

        return result

    def _step1_social_reasoning(
        self, instruction: str, context: Optional[str] = None
    ) -> Dict[str, str]:
        """Reason about human social norms and generate response."""
        prompt = f"""You are an expert in human social behavior and non-verbal communication.

Instruction: {instruction}
{f"Context: {context}" if context else ""}

Please provide:
1. Reasoning about relevant social norms and expectations
2. A detailed description of how a human would naturally respond to this situation, including:
   - Body language
   - Facial expressions (if applicable)
   - Gestures and movements
   - Timing and pace of actions

Format your response as JSON with keys: "reasoning" and "behavior"
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in human social behavior and non-verbal communication. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        result = json.loads(response.choices[0].message.content)
        return result

    def _step2_robot_translation(
        self,
        instruction: str,
        social_response: Dict[str, str],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Translate human behavior to robot-specific procedures."""
        constraint_desc = ""
        if constraints:
            constraint_desc = (
                f"\n\nRobot Constraints:\n{json.dumps(constraints, indent=2)}"
            )

        prompt = f"""You are a robotics expert specializing in humanoid robot control.

Original Instruction: {instruction}

Human Behavior Description:
{social_response["behavior"]}

Robot Capabilities:
- Unitree G1 humanoid robot with articulated arms
- Available joints: shoulders (3 DOF each), elbows (1 DOF each)
- Can move base (translation and rotation)
- Pre-defined reference motions: wave, point, nod
- Smooth interpolation between poses
- Timing control for synchronized movements

Available Default Poses:
{json.dumps(self.DEFAULT_POSES, indent=2)}

{constraint_desc}

Translate the human behavior into a detailed robot procedure. Include:
1. Which joints to actuate
2. Target poses or joint angles
3. Timing and sequencing
4. Any base movements needed
5. References to existing motions if applicable

Format your response as JSON with the following structure:
{{
    "motion_sequence": [
        {{
            "action": "description",
            "joints": ["joint_name", ...],
            "target_pose": "pose_name or custom angles",
            "duration": 1.0,
            "timing_offset": 0.0
        }},
        ...
    ],
    "base_movement": {{
        "translation": [x, y],
        "rotation": yaw_angle
    }},
    "overall_duration": total_time
}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a robotics expert. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )

        result = json.loads(response.choices[0].message.content)
        return result

    def _step3_code_generation(
        self,
        instruction: str,
        robot_procedure: Dict[str, Any],
        social_response: Dict[str, str],
    ) -> Dict[str, Any]:
        """Generate executable Python code for the motion."""
        example_code = '''
def wave_motion(t, qpos):
    """Animate right arm wave."""
    qpos[26] = 0.855
    qpos[27] = -0.611
    qpos[28] = -0.244
    qpos[29] = -1.75
    
    wave_start = 0.5
    wave_duration = 2.0
    
    if t < wave_start:
        qpos[22] = 0.75
        qpos[23] = -0.558
        qpos[24] = -0.489
        qpos[25] = -1.75
    elif t < wave_start + wave_duration:
        phase = 2 * np.pi * 2.0 * (t - wave_start)
        qpos[22] = -1.26
        qpos[23] = -0.157
        qpos[24] = 0.96
        qpos[25] = -0.7 + 0.3 * np.sin(phase)
    
    return qpos
'''

        prompt = f"""You are an expert Python programmer specializing in robot control code.

Generate a motion function based on this robot procedure:

Instruction: {instruction}
Social Context: {social_response["behavior"]}

Robot Procedure:
{json.dumps(robot_procedure, indent=2)}

Joint Indices:
{json.dumps(self.JOINT_INDICES, indent=2)}

Default Poses:
{json.dumps(self.DEFAULT_POSES, indent=2)}

Example:
{example_code}

Generate a function that:
1. Takes parameters: t (time in seconds), qpos (numpy array of joint positions)
2. Returns modified qpos
3. Uses smooth interpolation for natural motion
4. Follows the timing specified in the procedure
5. Has a descriptive function name based on the instruction
6. Includes docstring

IMPORTANT:
- Use numpy for calculations (imported as np)
- Joint indices are fixed (see JOINT_INDICES above)
- Always set ALL joints (left and right arms)
- Use smooth interpolation functions like: s = 3*s**2 - 2*s**3 for smoothstep
- The qpos array has indices 0-6 for base (pos + quat), then joint angles starting at index 7

Format your response as JSON with keys:
- function_name: name of the function
- code: function code as string
- parameters: dict of configurable parameters
- timing: dict with start_time and duration"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python programmer. Always respond with valid JSON. Generate clean, well-documented code.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )

        result = json.loads(response.choices[0].message.content)
        return result

    def refine_with_feedback(self, function_name: str, feedback: str) -> Dict[str, Any]:
        """Refine generated code based on user feedback."""
        if function_name not in self.skill_library:
            raise ValueError(f"Function '{function_name}' not found in skill library")

        original = self.skill_library[function_name]

        prompt = f"""You are refining robot motion code based on user feedback.

Original Instruction: {original["instruction"]}
User Feedback: {feedback}

Current Code:
{original["generated_code"]}

Current Parameters:
{json.dumps(original.get("parameters", {}), indent=2)}

Please update the code to incorporate the feedback. Determine if this is:
1. A high-level behavioral change (requires new procedure)
2. A low-level parameter adjustment (modify existing code)

Format your response as JSON with keys:
- "change_type": "behavioral" or "parameter"
- "updated_code": the modified function code
- "parameters": updated parameter dictionary
- "explanation": brief explanation of changes made
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at refining robot code. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )

        result = json.loads(response.choices[0].message.content)

        original["generated_code"] = result["updated_code"]
        original["parameters"] = result["parameters"]
        original["feedback_history"] = original.get("feedback_history", [])
        original["feedback_history"].append(
            {"feedback": feedback, "explanation": result["explanation"]}
        )

        return original

    def _add_to_skill_library(self, result: Dict[str, Any]):
        """Add motion to skill library."""
        self.skill_library[result["function_name"]] = result

    def get_skill_library(self) -> Dict[str, Dict[str, Any]]:
        """Return skill library."""
        return self.skill_library

    def export_skill(self, function_name: str, output_path: str):
        """Export skill to Python file."""
        if function_name not in self.skill_library:
            raise ValueError(f"Function '{function_name}' not found in skill library")

        skill = self.skill_library[function_name]

        code = f'''"""
{function_name}: {skill["instruction"]}
"""

import numpy as np

{skill["generated_code"]}

PARAMETERS = {json.dumps(skill.get("parameters", {}), indent=2)}
TIMING = {json.dumps(skill.get("timing", {}), indent=2)}
'''

        with open(output_path, "w") as f:
            f.write(code)

    def load_skill_from_file(self, file_path: str, function_name: str):
        """Load skill from file."""
        with open(file_path, "r") as f:
            code = f.read()

        self.skill_library[function_name] = {
            "function_name": function_name,
            "generated_code": code,
            "loaded_from_file": file_path,
        }

    def generate_scenario(
        self, instructions: List[Tuple[str, float]], output_path: str
    ):
        """Generate complete scenario file with multiple sequenced motions."""
        motions = []
        for instruction, start_time in instructions:
            result = self.generate_expressive_motion(instruction)
            motions.append({"result": result, "start_time": start_time})

        imports = "import numpy as np\nimport mujoco\nfrom pathlib import Path\n\n"
        functions_code = "\n\n".join([m["result"]["generated_code"] for m in motions])

        dispatcher_code = "def dispatch_motion(t, qpos):\n"

        for i, motion in enumerate(motions):
            start = motion["start_time"]
            func_name = motion["result"]["function_name"]
            duration = motion["result"].get("timing", {}).get("duration", 5.0)
            end = start + duration

            if i == 0:
                dispatcher_code += f"    if t < {end}:\n"
            else:
                dispatcher_code += f"    elif t < {end}:\n"
            dispatcher_code += f"        qpos = {func_name}(t - {start}, qpos)\n"

        dispatcher_code += "    return qpos\n"

        simulation_code = """
def yaw_to_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def run_simulation():
    path = Path('xmls/scene.xml')
    model = mujoco.MjModel.from_xml_string(path.read_text())
    data = mujoco.MjData(model)
    
    import mujoco_viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    hz = 50
    
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

if __name__ == "__main__":
    run_simulation()
"""

        full_code = (
            imports
            + functions_code
            + "\n\n"
            + dispatcher_code
            + "\n\n"
            + simulation_code
        )

        with open(output_path, "w") as f:
            f.write(full_code)

        print(f"Scenario saved to: {output_path}")
        return full_code


def main():
    """Example usage."""
    controller = LLMMotionController()

    result = controller.generate_expressive_motion(
        instruction="Acknowledge the person walking by",
        context="A person is walking past the robot in a hallway",
    )

    print(f"\nFunction Name: {result['function_name']}")
    print(f"\nSocial Reasoning:\n{result['social_reasoning']}")
    print(f"\nHuman Behavior:\n{result['human_behavior']}")
    print(f"\nRobot Procedure:\n{json.dumps(result['robot_procedure'], indent=2)}")
    print(f"\nGenerated Code:\n{result['generated_code']}")

    refined = controller.refine_with_feedback(
        function_name=result["function_name"],
        feedback="Make the wave slower and add a slight head nod",
    )

    print(f"\nRefined Code:\n{refined['generated_code']}")

    controller.export_skill(result["function_name"], "generated_acknowledge_motion.py")
    print("\nExported to: generated_acknowledge_motion.py")

    scenario_instructions = [
        ("Wave to greet someone", 0.0),
        ("Point at an object on the table", 3.0),
        ("Nod in agreement", 6.0),
    ]

    controller.generate_scenario(scenario_instructions, "eval/generated_scenario.py")
    print(f"\nComplete. {len(controller.get_skill_library())} skills in library")


if __name__ == "__main__":
    main()
