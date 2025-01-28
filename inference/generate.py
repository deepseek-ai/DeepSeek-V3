import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    try:
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
    except Exception as e:
        print(f"Erro ao amostrar token: {e}")
        raise


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    try:
        prompt_lens = [len(t) for t in prompt_tokens]
        assert max(prompt_lens) <= model.max_seq_len, f"Prompt é maior do que o comprimento máximo do modelo ({model.max_seq_len})"
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
    except Exception as e:
        print(f"Erro ao gerar texto: {e}")
        raise


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    try:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"O caminho para o checkpoint '{ckpt_path}' não existe.")
        
        if not os.path.exists(config):
            raise FileNotFoundError(f"O arquivo de configuração '{config}' não foi encontrado.")
        
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
        if world_size > 1:
            dist.init_process_group("nccl")
        
        global print
        if rank != 0:
            print = lambda *_, **__: None
        
        torch.cuda.set_device(local_rank)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(8)
        torch.manual_seed(965)
        
        with open(config) as f:
            args = ModelArgs(**json.load(f))
        print(args)
        
        with torch.device("cuda"):
            model = Transformer(args)
        
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        
        tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
        
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        
        if interactive:
            messages = []
            while True:
                try:
                    if world_size == 1:
                        prompt = input(">>> ")
                    elif rank == 0:
                        prompt = input(">>> ")
                        objects = [prompt]
                        dist.broadcast_object_list(objects, 0)
                    else:
                        objects = [None]
                        dist.broadcast_object_list(objects, 0)
                        prompt = objects[0]
                    
                    if prompt == "/exit":
                        break
                    elif prompt == "/clear":
                        messages.clear()
                        continue
                    
                    messages.append({"role": "user", "content": prompt})
                    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                    completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
                    completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
                    print(completion)
                    messages.append({"role": "assistant", "content": completion})
                
                except Exception as e:
                    print(f"Erro durante a interação: {e}")
                    continue
        
        else:
            with open(input_file) as f:
                prompts = [line.strip() for line in f.readlines()]
            
            assert len(prompts) <= args.max_batch_size
            prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
            completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
            completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
            
            for prompt, completion in zip(prompts, completions):
                print("Prompt:", prompt)
                print("Completion:", completion)
                print()
    
    except FileNotFoundError as e:
        print(f"Erro de arquivo: {e}")
    
    except Exception as e:
        print(f"Erro na execução principal: {e}")
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        parser = ArgumentParser()
        parser.add_argument("--ckpt-path", type=str, required=True)
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--input-file", type=str, default="")
        parser.add_argument("--interactive", action="store_true")
        parser.add_argument("--max-new-tokens", type=int, default=200)
        parser.add_argument("--temperature", type=float, default=0.2)
        args = parser.parse_args()
        assert args.input_file or args.interactive
        main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
    except Exception as e:
        print(f"Erro na execução do script: {e}")
