import os
import json
from argparse import ArgumentParser
from typing import List, Optional
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model
from model import Transformer, ModelArgs

def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
    if temperature <= 1e-5:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove_mask = cum_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        remove_indices = remove_mask.scatter(-1, sorted_indices, remove_mask)
        logits[remove_indices] = -float('Inf')
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
    return (logits + gumbel_noise).argmax(dim=-1)

@torch.inference_mode()
def generate(model: Transformer, prompt_tokens: List[List[int]], max_new_tokens: int, eos_id: int, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None) -> List[List[int]]:
    model.reset_cache()
    prompt_lens = [len(t) for t in prompt_tokens]
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, device="cuda")
    prev_pos = 0
    finished = torch.zeros(len(prompt_tokens), dtype=torch.bool, device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = sample(logits, temperature, top_k, top_p)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= (~prompt_mask[:, cur_pos] & (next_token == eos_id))
        prev_pos = cur_pos
        if finished.all():
            break
    completions = []
    for i, seq in enumerate(tokens.tolist()):
        seq = seq[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        completions.append(seq[:seq.index(eos_id)] if eos_id in seq else seq)
    return completions

def main(ckpt_path: str, config: str, input_file: str = "", interactive: bool = True, max_new_tokens: int = 100, temperature: float = 0.2, top_k: Optional[int] = None, top_p: Optional[float] = None) -> None:
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_path}")
    if not os.path.isfile(config):
        raise FileNotFoundError(f"Config file missing: {config}")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.manual_seed(965)
    with open(config) as f:
        model_args = ModelArgs(**json.load(f))
    model = Transformer(model_args).to(torch.bfloat16).cuda()
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if interactive:
        messages = []
        while True:
            prompt = get_input(rank, world_size)
            if prompt == "/exit":
                break
            if prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            try:
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            except Exception as e:
                print(f"Tokenization error: {e}")
                continue
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature, top_k, top_p)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        batch_size = model_args.max_batch_size
        completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True) for p in batch_prompts]
            completion_tokens = generate(model, batch_tokens, max_new_tokens, tokenizer.eos_token_id, temperature, top_k, top_p)
            completions.extend(tokenizer.batch_decode(completion_tokens, skip_special_tokens=True))
        for prompt, completion in zip(prompts, completions):
            print(f"Prompt: {prompt}\nCompletion: {completion}\n{'-'*50}")
    if world_size > 1:
        dist.destroy_process_group()

def get_input(rank: int, world_size: int) -> str:
    if world_size == 1 or rank == 0:
        prompt = input(">>> ")
        if world_size > 1:
            dist.broadcast_object_list([prompt], src=0)
        return prompt
    else:
        res = [None]
        dist.broadcast_object_list(res, src=0)
        return res[0]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
