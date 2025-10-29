"""Simple example of using the motion controller."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.llm_motion_controller import LLMMotionController


def main():
    # Use 'mini' (default, cheapest), 'turbo', 'gpt4o', or 'gpt4'
    controller = LLMMotionController(model="mini")

    result = controller.generate_expressive_motion(
        instruction="Acknowledge the person walking by",
        context="A person is walking past the robot in a hallway",
    )

    print(f"Generated: {result['function_name']}")
    print(f"\nReasoning: {result['social_reasoning'][:150]}...")
    print(f"\nCode preview:\n{result['generated_code'][:300]}...")

    controller.export_skill(result["function_name"], "generated_motion.py")
    print("\nExported to: generated_motion.py")

    refined = controller.refine_with_feedback(
        function_name=result["function_name"], feedback="Make the wave slower"
    )
    print("\nRefined based on feedback")


if __name__ == "__main__":
    main()
