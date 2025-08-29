"""
DeepSeek-V3 Function Calling Demo

This script demonstrates how to use function calling with the DeepSeek-V3 model
after applying the fix for the chat template.
"""

from transformers import AutoTokenizer
import json
import argparse


def get_current_temperature(location: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: Get the location of the temperature in the format of "city, country"
    Returns:
        Displays the current temperature at the specified location as a floating point number.
    """
    # This is a mock function that would normally call a weather API
    print(f"Getting temperature for {location}")
    return 22.0


def get_current_time(timezone: str) -> str:
    """
    Get the current time in a specific timezone.

    Args:
        timezone: The timezone to get the current time for (e.g., "UTC", "America/New_York")
    Returns:
        The current time as a string.
    """
    # This is a mock function that would normally get the current time
    print(f"Getting time for timezone {timezone}")
    return "12:30 PM"


def main():
    parser = argparse.ArgumentParser(
        description="Test DeepSeek-V3 function calling")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the DeepSeek-V3 model")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)

    # Example 1: Simple weather query
    print("\nExample 1: Weather query")
    tool_call = {"name": "get_current_temperature",
                 "arguments": {"location": "Paris, France"}}
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides weather information."},
        {"role": "user", "content": "What is the temperature in Paris now?"},
        {"role": "assistant", "tool_calls": [
            {"type": "function", "function": tool_call}]},
        {"role": "user", "content": "Thanks for checking! Is that warm for this time of year?"}
    ]

    print("Processing chat template...")
    chat_input = tokenizer.apply_chat_template(
        messages,
        tools=[get_current_temperature],
        add_generation_prompt=True,
        tokenize=False,
        tools_in_user_message=False
    )

    print("\nGenerated chat template:")
    print("-" * 50)
    print(chat_input)
    print("-" * 50)

    # Example 2: Multiple tool calls
    print("\nExample 2: Multiple function calls")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the time in New York and temperature in Tokyo?"},
        {"role": "assistant", "tool_calls": [
            {"type": "function", "function": {"name": "get_current_time",
                                              "arguments": {"timezone": "America/New_York"}}},
            {"type": "function", "function": {
                "name": "get_current_temperature", "arguments": {"location": "Tokyo, Japan"}}}
        ]},
    ]

    print("Processing chat template...")
    chat_input = tokenizer.apply_chat_template(
        messages,
        tools=[get_current_time, get_current_temperature],
        add_generation_prompt=True,
        tokenize=False,
        tools_in_user_message=False
    )

    print("\nGenerated chat template:")
    print("-" * 50)
    print(chat_input)
    print("-" * 50)


if __name__ == "__main__":
    main()
