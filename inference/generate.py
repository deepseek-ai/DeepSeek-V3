import os
import json
import logging
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, "Prompt length exceeds model max sequence length"
    
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = sample(logits, temperature) if temperature > 0 else logits.argmax(dim=-1)
        
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        
        prev_pos = cur_pos
        if finished.all():
            break
    
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        dist.init_process_group("nccl")
    
    if rank != 0:
        logger.setLevel(logging.WARNING)
    
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    
    # Load model args
    try:
        with open(config) as f:
            args = ModelArgs(**json.load(f))
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file: {e}")
        return
    
    logger.info(f"Model args: {args}")
    
    # Load the model on GPU
    with torch.device("cuda"):
        model = Transformer(args)
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # Generate a test sequence to verify everything is working
    test_prompt = "DeepSeek"
    test_tokens = tokenizer.encode(test_prompt)
    generated_tokens = generate(model, [test_tokens], 2, tokenizer.eos_token_id, 1.0)
    logger.info(f"Generated test output: {tokenizer.decode(generated_tokens[0])}")
    
    # Load model weights
    try:
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Interactive mode or batch processing
    if interactive:
        messages = []
        while True:
            if world_size == 1 or rank == 0:
                prompt = input(">>> ")
                if prompt == "/exit":
                    break
                elif prompt == "/clear":
                    messages.clear()
                    continue
                messages.append({"role": "user", "content": prompt})
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
                completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
                logger.info(f"Generated completion: {completion}")
                messages.append({"role": "assistant", "content": completion})
            elif rank != 0:
                # Synchronize input across multiple nodes
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
    
    else:
        # Batch processing mode
        if not input_file:
            logger.error("Input file is required for batch processing mode")
            return
        try:
            with open(input_file) as f:
                prompts = [line.strip() for line in f.readlines()]
        except FileNotFoundError as e:
            logger.error(f"Input file not found: {e}")
            return
        
        assert len(prompts) <= args.max_batch_size, "Exceeds batch size limit"
        
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("--input-file", type=str, default="", help="File containing prompts for batch processing.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling.")
    
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
