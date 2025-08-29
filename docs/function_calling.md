# Function Calling with DeepSeek-V3

This document provides guidance on using function calling with DeepSeek-V3 models.

## Overview

Function calling allows the model to call external functions through a structured interface. It's particularly useful for:

- Retrieving real-time information (weather, time, data from APIs)
- Performing calculations
- Executing actions based on user requests

## Usage with Transformers

DeepSeek-V3 supports function calling through the Hugging Face Transformers library. The example below demonstrates how to use this feature:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define your function
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    # In a real application, this would call a weather API
    return f"Sunny, 22°C in {location}"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)

# Create a conversation with function calling
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in Tokyo?"},
    {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Tokyo, Japan"}}}
    ]},
    {"role": "user", "content": "Thanks! And what about New York?"}
]

# Apply the chat template
inputs = tokenizer.apply_chat_template(
    messages, 
    tools=[get_weather],
    add_generation_prompt=True, 
    tokenize=True,
    tools_in_user_message=False
)

# Generate a response
output_ids = model.generate(inputs, max_new_tokens=100)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
```

## Function Definitions

Functions must have type annotations and docstrings following the OpenAI format:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    Returns:
        Description of what is returned
    """
    # Function implementation
    pass
```

## Limitations

- Function parameters must be JSON-serializable types
- Function return values should also be JSON-serializable
- Complex object types are not directly supported

For more advanced use cases, please refer to the Hugging Face documentation on function calling.