import os
import json
from argparse import ArgumentParser
from typing import List, Optional

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


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
    temperature: float = 1.0,
    device: str = "mps",
    chunk_size: int = 512
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.
    Optimized for MacBook with chunked processing and reduced memory usage.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
        device (str, optional): The device to run generation on. Defaults to "mps".
        chunk_size (int, optional): Size of processing chunks for memory efficiency. Defaults to 512.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    
    # Process in smaller batches for memory efficiency
    batch_size = len(prompt_tokens)
    if batch_size > 1:
        print(f"Processing {batch_size} prompts in sequence to conserve memory...")
        all_completion_tokens = []
        for i in range(batch_size):
            single_completion = generate(
                model, 
                [prompt_tokens[i]], 
                max_new_tokens, 
                eos_id, 
                temperature,
                device,
                chunk_size
            )
            all_completion_tokens.extend(single_completion)
        return all_completion_tokens
    
    # Calculate total sequence length
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    
    # Initialize tokens tensor on appropriate device
    tokens = torch.full((batch_size, total_len), -1, dtype=torch.long, device=device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    
    prev_pos = 0
    finished = torch.tensor([False] * batch_size, device=device)
    prompt_mask = tokens != -1
    
    # Process in chunks for lower memory usage
    for cur_pos in range(min(prompt_lens), total_len):
        # Use a sliding window approach for KV cache efficiency
        start_pos = max(0, cur_pos - chunk_size) if cur_pos > min(prompt_lens) else prev_pos
        
        # Clear GPU/MPS cache periodically to prevent memory fragmentation
        if cur_pos % 50 == 0 and device == "mps":
            torch.mps.empty_cache()
        elif cur_pos % 50 == 0 and device == "cuda":
            torch.cuda.empty_cache()
            
        logits = model.forward(tokens[:, start_pos:cur_pos], start_pos)
        
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
            
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = start_pos
        
        if finished.all():
            break
            
        # Optional progress display for longer generations
        if max_new_tokens > 100 and cur_pos % 20 == 0:
            progress = (cur_pos - min(prompt_lens)) / max_new_tokens * 100
            print(f"\rGenerating: {progress:.1f}% complete", end="")
    
    if max_new_tokens > 100:
        print("\rGeneration complete!          ")
        
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
        
    return completion_tokens


def get_optimal_device(force_cpu: bool = False) -> str:
    """
    Determines the best available device for model inference.
    
    Args:
        force_cpu (bool, optional): Force CPU usage even if GPU is available. Defaults to False.
        
    Returns:
        str: Device string ("cuda", "mps", or "cpu")
    """
    if force_cpu:
        return "cpu"
        
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"


def optimize_model(model: Transformer, quantize: bool = True, device: str = "cpu") -> Transformer:
    """
    Applies optimizations to the model for more efficient inference.
    
    Args:
        model (Transformer): The transformer model to optimize.
        quantize (bool, optional): Whether to apply int8 quantization. Defaults to True.
        device (str, optional): Target device. Defaults to "cpu".
        
    Returns:
        Transformer: Optimized model
    """
    if quantize and device == "cpu":
        # Apply dynamic quantization to linear layers
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    return model


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    force_cpu: bool = False,
    quantize: bool = True,
    chunk_size: int = 512,
    reduced_model: bool = False,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.
    Optimized for MacBooks with various memory/performance options.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
        force_cpu (bool, optional): Force CPU usage even if GPU is available. Defaults to False.
        quantize (bool, optional): Apply quantization when possible. Defaults to True.
        chunk_size (int, optional): Size of processing chunks for memory efficiency. Defaults to 512.
        reduced_model (bool, optional): Load a smaller version of the model if available. Defaults to False.
    """
    # Detect optimal device
    device = get_optimal_device(force_cpu)
    print(f"Using device: {device}")
    
    # Set appropriate torch settings
    if device == "cuda":
        torch.cuda.set_device(0)
    
    # Use bfloat16 for CUDA/MPS, float32 for CPU
    if device == "cpu":
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.bfloat16)
    
    torch.set_num_threads(8)  # Adjust based on your CPU
    torch.manual_seed(965)
    
    # Load model configuration
    with open(config) as f:
        config_data = json.load(f)
        
        # Apply optimizations to configuration for smaller/faster model
        if reduced_model:
            # Reduce the number of experts and heads
            config_data["n_routed_experts"] = config_data.get("n_routed_experts", 64) // 2
            config_data["max_seq_len"] = min(config_data.get("max_seq_len", 16384), 4096)  # Reduce context size
            
        args = ModelArgs(**config_data)
    
    print(args)
    
    # Load model
    with torch.device(device):
        model = Transformer(args)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # Load the appropriate checkpoint
    if device != "cuda":
        # For CPU/MPS, always use rank 0
        checkpoint_path = os.path.join(ckpt_path, "model0-mp1.safetensors")
    else:
        # For CUDA, can use multiple GPUs if available
        world_size = min(torch.cuda.device_count(), 1)  # Limit to 1 for simpler usage
        rank = 0
        checkpoint_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    load_model(model, checkpoint_path)
    
    # Apply quantization and other optimizations
    model = optimize_model(model, quantize=quantize, device=device)
    model.to(device)
    
    # Generate a quick test sequence to ensure everything is working
    print("Running warmup generation...")
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1., device)[0])
    print("Model loaded and ready!")

    if interactive:
        messages = []
        while True:
            prompt = input(">>> ")
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
                
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            # Show a waiting message for longer generations
            if max_new_tokens > 50:
                print("Generating response...")
                
            completion_tokens = generate(
                model, 
                [prompt_tokens], 
                max_new_tokens, 
                tokenizer.eos_token_id, 
                temperature,
                device,
                chunk_size
            )
            
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
            
            # Clear cache after each generation to prevent memory buildup
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(
            model, 
            prompt_tokens, 
            max_new_tokens, 
            tokenizer.eos_token_id, 
            temperature,
            device,
            chunk_size
        )
        
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()


if __name__ == "__main__":
    """
    Command-line interface for optimized text generation on MacBooks.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.
        --force-cpu (bool, optional): Force CPU usage even if GPU is available.
        --no-quantize (bool, optional): Disable quantization (higher quality but slower).
        --chunk-size (int, optional): Size of processing chunks for memory efficiency. Defaults to 512.
        --reduced-model (bool, optional): Load a smaller version of the model if available.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--reduced-model", action="store_true")
    
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    
    main(
        args.ckpt_path, 
        args.config, 
        args.input_file, 
        args.interactive, 
        args.max_new_tokens, 
        args.temperature,
        args.force_cpu,
        not args.no_quantize,
        args.chunk_size,
        args.reduced_model
    )
