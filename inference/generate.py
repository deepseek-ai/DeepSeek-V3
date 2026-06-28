import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

# Attempt to import local store; if unavailable, local memory feature is disabled
try:
    from local_store import LocalStore
    LOCAL_STORE_AVAILABLE = True
except ImportError:
    LOCAL_STORE_AVAILABLE = False


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(tokens) for tokens in prompt_tokens]
    assert model.args.max_seq_len >= max(prompt_lens) + max_new_tokens, \
        f"Prompt length {max(prompt_lens)} + max_new_tokens {max_new_tokens} exceeds max_seq_len {model.args.max_seq_len}"

    # Prepare input tensors
    total_len = max(prompt_lens) + max_new_tokens
    batch_size = len(prompt_tokens)
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    tokens = torch.full((batch_size, total_len), eos_id, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    input_mask = tokens != eos_id
    start_pos = 0
    prev_pos = 0

    # Pre-fill
    if min_prompt_len == max_prompt_len:
        logits = model.forward(tokens[:, :max_prompt_len], start_pos, input_mask[:, :max_prompt_len])
        start_pos = max_prompt_len
    else:
        logits = model.forward(tokens[:, :min_prompt_len], start_pos, input_mask[:, :min_prompt_len])
        start_pos = min_prompt_len

    # Decode
    for cur_pos in range(start_pos, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, input_mask[:, prev_pos:cur_pos])
        next_token = sample(logits, temperature)
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos

    return [tokens[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens].tolist() for i in range(batch_size)]


def main():
    parser = ArgumentParser(description="DeepSeek-V3 text generation with optional on-device persistent memory.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the model configuration file.")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json", help="Path to the tokenizer file.")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--eos-id", type=int, default=32078, help="End-of-sequence token ID.")
    parser.add_argument("--local-store-path", type=str, default=None,
                        help="Path to local storage directory for persistent memory. Enables on-device context.")
    parser.add_argument("--max-context-tokens", type=int, default=8192,
                        help="Maximum tokens to use for local context injection (default: 8192).")
    parser.add_argument("--save-context", action="store_true", default=True,
                        help="Save conversation context after generation (default: True).")
    parser.add_argument("--no-save-context", action="store_false", dest="save_context",
                        help="Disable saving context after generation.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Build model
    print("Loading model...")
    model = Transformer(model_args)
    load_model(model, os.path.join(args.ckpt_path, "model.safetensors"))
    print("Model loaded.")

    # Initialize local store if requested
    local_store = None
    if args.local_store_path and LOCAL_STORE_AVAILABLE:
        print(f"Initializing local memory store at: {args.local_store_path}")
        local_store = LocalStore(args.local_store_path)
        # Load existing context
        context = local_store.load_context()
        if context:
            print("Loaded previous context from local store.")
            # Build system prompt from context
            sys_parts = []
            if 'profile' in context and context['profile']:
                sys_parts.append(f"[User Profile: {json.dumps(context['profile'])}]")
            if 'summaries' in context and context['summaries']:
                # Use the last summary
                sys_parts.append(f"[Conversation Summary: {context['summaries'][-1]}]")
            # Add identity hint
            identity_key = local_store.get_identity_key()
            if identity_key:
                short_id = identity_key[:8]
                sys_parts.append(f"[Identity: {short_id}]")
            if sys_parts:
                sys_prompt = "\n".join(sys_parts) + "\n"
                # Prepend system prompt to user prompt
                args.prompt = sys_prompt + args.prompt
        else:
            print("No stored context found; starting fresh session.")
    elif args.local_store_path and not LOCAL_STORE_AVAILABLE:
        print("Warning: --local-store-path specified but local_store module not available. Install 'cryptography'.")

    # Tokenize prompt
    prompt_tokens = [tokenizer.encode(args.prompt, add_special_tokens=False)]

    # Truncate prompt to max context tokens if needed
    max_prompt_len = model.args.max_seq_len - args.max_new_tokens
    if len(prompt_tokens[0]) > max_prompt_len:
        print(f"Truncating prompt from {len(prompt_tokens[0])} to {max_prompt_len} tokens.")
        prompt_tokens[0] = prompt_tokens[0][-max_prompt_len:]

    # Generate
    print("Generating...")
    generated_tokens = generate(
        model, prompt_tokens, args.max_new_tokens, args.eos_id, args.temperature
    )
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("\nGenerated response:")
    print(response)

    # Save conversation context if local store is active and save_context is True
    if local_store and args.save_context:
        # Build context from current conversation
        messages = [
            {"role": "user", "content": args.prompt},
            {"role": "assistant", "content": response}
        ]
        # Load existing context to preserve prior messages
        existing = local_store.load_context()
        if existing and "messages" in existing:
            all_messages = existing["messages"] + messages
            # Keep only last 20 message pairs to limit size
            if len(all_messages) > 40:
                all_messages = all_messages[-40:]
        else:
            all_messages = messages
        
        # Create a simple summary from the last exchange (placeholder for real summarization)
        summary_text = f"User said: {args.prompt[:100]}... ; Assistant replied: {response[:100]}..."
        
        new_context = {
            "messages": all_messages,
            "summaries": [summary_text],
            "profile": existing.get("profile", {}) if existing else {}
        }
        local_store.save_context(new_context)
        print("Conversation context saved to local store.")


if __name__ == "__main__":
    main()
