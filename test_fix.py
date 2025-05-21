from transformers import AutoTokenizer
import json
import os
import sys

# Function to test


def get_current_temperature(location: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: Get the location of the temperature in the format of "city, country"
    Returns:
        Displays the current temperature at the specified location as a floating point number (in the specified unit).
    """
    return 22.0


def test_with_original_tokenizer(model_path):
    print("Testing with original tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        tool_call = {"name": "get_current_temperature",
                     "arguments": {"location": "Paris, France"}}
        messages = [
            {"role": "system", "content": "You are a robot that responds to weather queries."},
            {"role": "user", "content": "What is the temperature in Paris now?"},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": tool_call}]},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tools=[get_current_temperature],
            add_generation_prompt=False,
            tokenize=False,
            tools_in_user_message=False
        )
        print("Success with original tokenizer!")
        print(inputs)
        return True
    except Exception as e:
        print(f"Error with original tokenizer: {e}")
        return False


def test_with_fixed_tokenizer(model_path, fixed_config_path):
    print("Testing with fixed tokenizer config...")
    try:
        # Read the original tokenizer files
        tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(tokenizer_json_path):
            print(f"Error: tokenizer.json not found at {tokenizer_json_path}")
            return False

        # Copy the tokenizer.json and use our fixed config
        fixed_dir = "fixed_tokenizer"
        os.makedirs(fixed_dir, exist_ok=True)

        # Copy tokenizer.json
        import shutil
        shutil.copy(tokenizer_json_path, os.path.join(
            fixed_dir, "tokenizer.json"))

        # Create fixed tokenizer_config.json
        with open(fixed_config_path, 'r') as f:
            fixed_config = f.read()

        with open(os.path.join(fixed_dir, "tokenizer_config.json"), 'w') as f:
            f.write(fixed_config)

        # Load the fixed tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            fixed_dir, trust_remote_code=True)

        tool_call = {"name": "get_current_temperature",
                     "arguments": {"location": "Paris, France"}}
        messages = [
            {"role": "system", "content": "You are a robot that responds to weather queries."},
            {"role": "user", "content": "What is the temperature in Paris now?"},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": tool_call}]},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tools=[get_current_temperature],
            add_generation_prompt=False,
            tokenize=False,
            tools_in_user_message=False
        )
        print("Success with fixed tokenizer!")
        print(inputs)
        return True
    except Exception as e:
        print(f"Error with fixed tokenizer: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_fix.py /path/to/DeepSeek-V3 /path/to/fixed_config.json")
        return

    model_path = sys.argv[1]
    fixed_config_path = sys.argv[2] if len(
        sys.argv) > 2 else "tokenizer_config.json"

    # Test with original tokenizer (should fail)
    original_success = test_with_original_tokenizer(model_path)

    # Test with fixed tokenizer (should succeed)
    fixed_success = test_with_fixed_tokenizer(model_path, fixed_config_path)

    if not original_success and fixed_success:
        print("\n✅ Fix was successful! The issue has been resolved.")
    else:
        print("\n❌ Testing did not confirm the fix. Please check the logs above.")


if __name__ == "__main__":
    main()
