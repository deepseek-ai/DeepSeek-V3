import os
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import timedelta
from contextlib import nullcontext

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
from tqdm import tqdm

from model import Transformer, ModelArgs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EOS_TOKEN = "</s>"
MAX_SEQ_LEN_WARNING_THRESHOLD = 0.9
TORCH_DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training environment."""
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(minutes=5)
        )
        logger.info(f"Initialized process group (rank {rank}/{world_size})")
    
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank

def validate_paths(ckpt_path: Path, config_path: Path) -> None:
    """Validate model checkpoint and config paths."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_path} not found")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")

def load_model_config(config_path: Path) -> ModelArgs:
    """Load and validate model configuration."""
    try:
        with open(config_path) as f:
            config_data = json.load(f)
        return ModelArgs(**config_data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Invalid model config: {str(e)}")
        raise

def initialize_model(args: ModelArgs, device: str) -> Transformer:
    """Initialize model with proper device placement and dtype."""
    model = Transformer(args).to(TORCH_DTYPE)
    model.eval()
    return model

def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
    """
    Sample token from logits with temperature and top-k filtering.
    
    Args:
        logits: Unnormalized log probabilities (batch_size, vocab_size)
        temperature: Sampling temperature (0.0 = greedy)
        top_k: Top-k tokens to consider (0 = no filtering)
    
    Returns:
        Sampled token indices (batch_size, 1)
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)

@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.1
) -> List[List[int]]:
    """
    Generate text with dynamic sequence length management.
    
    Args:
        model: Initialized transformer model
        prompt_tokens: List of tokenized prompts
        max_new_tokens: Maximum new tokens to generate
        eos_id: End-of-sequence token ID
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        List of generated token sequences
    """
    batch_size = len(prompt_tokens)
    device = next(model.parameters()).device
    max_seq_len = model.max_seq_len
    prompt_lens = [len(p) for p in prompt_tokens]
    
    # Adjust max_new_tokens based on input length
    if max(prompt_lens) + max_new_tokens > max_seq_len:
        logger.warning(f"Truncating sequence length to {max_seq_len}")
        max_new_tokens = max_seq_len - max(prompt_lens)

    # Initialize token tensor
    tokens = torch.full((batch_size, max_seq_len), -1, dtype=torch.long, device=device)
    for i, seq in enumerate(prompt_tokens):
        tokens[i, :len(seq)] = torch.tensor(seq, device=device)
    
    # Generation loop
    prev_pos = 0
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    prompt_mask = tokens != -1
    progress_bar = tqdm(total=max_new_tokens, desc="Generating", disable=not logger.isEnabledFor(logging.INFO))
    
    try:
        for cur_pos in range(max(prompt_lens), min(max_seq_len, max(prompt_lens) + max_new_tokens)):
            # Model forward pass
            logits = model(tokens[:, prev_pos:cur_pos], prev_pos)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for idx in range(batch_size):
                    unique_tokens, counts = torch.unique(tokens[idx], return_counts=True)
                    logits[idx, unique_tokens] /= counts.float() ** (repetition_penalty - 1.0)
            
            # Sample next tokens
            next_tokens = sample(logits[:, -1], temperature, top_k)
            
            # Update tokens
            tokens[:, cur_pos] = torch.where(
                prompt_mask[:, cur_pos],
                tokens[:, cur_pos],
                next_tokens
            )
            
            # Update completion status
            finished |= (~prompt_mask[:, cur_pos] & (next_tokens == eos_id))
            prev_pos = cur_pos
            progress_bar.update(1)
            
            if finished.all():
                break
    finally:
        progress_bar.close()
    
    # Process outputs
    return [seq[pl:pl + max_new_tokens].tolist() for pl, seq in zip(prompt_lens, tokens)]

def interactive_loop(
    model: Transformer,
    tokenizer: AutoTokenizer,
    world_size: int,
    rank: int,
    max_new_tokens: int,
    temperature: float
) -> None:
    """Interactive chat interface with history management."""
    messages = []
    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids(DEFAULT_EOS_TOKEN)
    
    while True:
        try:
            # Distributed input handling
            prompt = None
            if world_size > 1:
                if rank == 0:
                    prompt = input("\nUser: ")
                    dist.broadcast_object_list([prompt], src=0)
                else:
                    dist.broadcast_object_list([prompt], src=0)
            else:
                prompt = input("\nUser: ")
            
            # Command handling
            if prompt in ["/exit", "/clear"]:
                if prompt == "/exit":
                    break
                messages.clear()
                logger.info("History cleared")
                continue
            
            # Tokenize and generate
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                truncation=True,
                max_length=model.max_seq_len - max_new_tokens
            )
            
            completion_tokens = generate(
                model, 
                [prompt_tokens], 
                max_new_tokens, 
                eos_id, 
                temperature
            )[0]
            
            # Decode and update history
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": completion})
            print(f"\nAssistant: {completion}")
        
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            messages.pop()  # Remove failed prompt

def batch_process(
    model: Transformer,
    tokenizer: AutoTokenizer,
    input_file: Path,
    max_new_tokens: int,
    temperature: float
) -> None:
    """Batch processing mode with progress tracking."""
    try:
        with open(input_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if not prompts:
            raise ValueError("Input file is empty")
        
        # Tokenize with parallel processing
        tokenizer_fn = lambda p: tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], 
            add_generation_prompt=True,
            truncation=True,
            max_length=model.max_seq_len - max_new_tokens
        )
        prompt_tokens = [tokenizer_fn(p) for p in tqdm(prompts, desc="Tokenizing")]
        
        # Generate in batches
        completions = []
        for i in tqdm(range(0, len(prompt_tokens), model.args.max_batch_size)):
            batch = prompt_tokens[i:i + model.args.max_batch_size]
            completions += generate(model, batch, max_new_tokens, tokenizer.eos_token_id, temperature)
        
        # Decode and print
        for prompt, tokens in zip(prompts, completions):
            completion = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"\nPrompt: {prompt}\nCompletion: {completion}\n{'='*50}")
    
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise

def main(
    ckpt_path: str,
    config_path: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 200,
    temperature: float = 0.2
) -> None:
    """Main execution flow with proper resource management."""
    # Distributed setup
    world_size, rank, local_rank = setup_distributed()
    
    try:
        # Path validation
        ckpt_dir = Path(ckpt_path)
        config_file = Path(config_path)
        validate_paths(ckpt_dir, config_file)
        
        # Model initialization
        model_args = load_model_config(config_file)
        model = initialize_model(model_args, DEVICE)
        load_model(model, ckpt_dir / f"model{rank}-mp{world_size}.safetensors")
        
        # Tokenizer setup
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_dir,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Generation mode selection
        if interactive:
            interactive_loop(model, tokenizer, world_size, rank, max_new_tokens, temperature)
        else:
            batch_process(model, tokenizer, Path(input_file), max_new_tokens, temperature)
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == "__main__":
    parser = ArgumentParser(description="Distributed Transformer Text Generation")
    parser.add_argument("--ckpt-path", type=str, required=True,
                       help="Path to model checkpoint directory")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to model config JSON file")
    parser.add_argument("--input-file", type=str, default="",
                       help="Path to input file for batch processing")
    parser.add_argument("--interactive", action="store_true",
                       help="Enable interactive chat mode")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO",
                       help="Set logging verbosity")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.input_file:
        parser.error("Must specify either --interactive or --input-file")
    
    # Configure logging
    logger.setLevel(args.log_level)
    
    try:
        main(
            args.ckpt_path,
            args.config,
            args.input_file,
            args.interactive,
            args.max_new_tokens,
            args.temperature
        )
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        exit(1)
