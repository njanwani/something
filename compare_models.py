"""Compare different models for motion generation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.llm_motion_controller import LLMMotionController
import time


def test_model(model_name):
    print(f"\n{'=' * 60}")
    print(f"Testing: {model_name}")
    print("=" * 60)

    controller = LLMMotionController(model=model_name)

    start = time.time()
    result = controller.generate_expressive_motion(
        instruction="Wave hello", context="Someone just entered the room"
    )
    duration = time.time() - start

    print(f"Time: {duration:.1f}s")
    print(f"Function: {result['function_name']}")
    print(f"Code length: {len(result['generated_code'])} chars")
    print(f"Reasoning length: {len(result['social_reasoning'])} chars")

    return result


def main():
    """Test different models."""

    models = ["mini", "turbo"]  # Start with cheap ones

    print("Testing cheap models for motion generation...")
    print("Add 'gpt4o' or 'gpt4' to test higher-tier models")

    for model in models:
        try:
            test_model(model)
        except Exception as e:
            print(f"Error with {model}: {e}")

    print(f"\n{'=' * 60}")
    print("Recommendation: Use 'mini' for development")
    print("Switch to 'gpt4o' only if you need better quality")
    print("=" * 60)


if __name__ == "__main__":
    main()
