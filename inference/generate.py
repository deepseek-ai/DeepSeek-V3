import os
import json
import sys
from argparse import ArgumentParser
from typing import List, Generator

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0, top_p: float = 0.9):
    """
    使用温度缩放和核采样从 logits 中采样一个 token | Samples a token from the logits using temperature scaling and nucleus sampling.
    """
    # 如果温度为0，直接贪婪解码（确定性） | If temperature is 0, greedy decode directly (deterministic)
    if temperature == 0:
        return logits.argmax(dim=-1)
    
    logits = logits / max(temperature, 1e-5) # 应用温度缩放 | Apply temperature scaling
    probs = torch.softmax(logits, dim=-1) # 计算概率 | Calculate probabilities
    
    # 核采样 | Nucleus Sampling (Top-p)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 移除核之外的 token | Remove tokens outside the nucleus (cumulative > top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs.masked_fill_(indices_to_remove, 0.0)
        probs.div_(probs.sum(dim=-1, keepdim=True)) # 重新归一化 | Renormalize

    # 使用 Gumbel-Max 技巧进行采样 | Sample using the Gumbel-Max trick
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def apply_repetition_penalty(logits, tokens, penalty=1.0):
    """
    对已经出现过的 token 应用惩罚以避免重复循环 | Applies penalty to tokens that have already appeared to avoid repetition loops.
    """
    if penalty <= 1.0: # 如果惩罚未激活则跳过 | Skip if penalty is not active
        return logits
    
    # 为重复的 token 创建掩码 | Create a mask for repeated tokens
    # 简化：如果 token 在历史记录中，则应用惩罚 | Simplification: apply penalty if the token is in history
    # 针对小批次的优化 | Optimized for small batches
    batch_size = logits.shape[0]
    for i in range(batch_size):
        # 获取目前为止生成的唯一 token | Get unique tokens generated so far
        seen_tokens = tokens[i].unique()
        # 惩罚这些索引上的 logits | Penalize logits at these indices
        logits[i].index_put_((seen_tokens,), logits[i][seen_tokens] / penalty)
    
    return logits


@torch.inference_mode()
def generate_stream(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    stop_tokens: List[int],
    temperature: float = 1.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0
) -> Generator[List[int], None, None]:
    """
    逐个生成 token (yield) 以允许流式传输 | Generates tokens one by one (yield) to allow streaming.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"提示长度超过模型最大序列长度"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    
    device = "cuda"
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=device)
    prompt_mask = tokens != -1

    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        
        # 在采样之前应用重复惩罚 | Apply repetition penalty BEFORE sampling
        if repetition_penalty > 1.0:
            logits = apply_repetition_penalty(logits[:, -1, :], tokens[:, :cur_pos], repetition_penalty)
        else:
            logits = logits[:, -1, :]

        if temperature > 0:
            next_token = sample(logits, temperature, top_p)
        else:
            next_token = logits.argmax(dim=-1)

        # 如果处于填充阶段，强制使用提示词 token | Force prompt tokens if in the filling phase
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token

        # 检查 eos 和停止 token | Check for eos and stop tokens
        is_eos = next_token == eos_id
        is_stop = False
        if stop_tokens:
            for st in stop_tokens:
                is_stop |= (next_token == st)

        finished |= torch.logical_and(~prompt_mask[:, cur_pos], is_eos | is_stop)

        # 仅为每个序列生成新 token | Yield only the new tokens for each sequence
        if not finished.all():
            yield next_token.tolist()

        prev_pos = cur_pos
        if finished.all():
            break


def load_model_components(ckpt_path: str, config_path: str, rank: int, world_size: int):
    """加载模型、分词器并配置数据类型。| Loads model, tokenizer and configures dtype."""
    
    # 检测 bfloat16 支持 | Detect bfloat16 support
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if rank == 0:
        print(f"使用 dtype: {dtype}")

    torch.set_default_dtype(dtype)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    
    if rank == 0:
        print(f"模型参数: {args}")

    with torch.device("cuda"):
        model = Transformer(args)

    # 优化的预热（同步且终端无垃圾信息）| Optimized warm-up (synchronization without terminal clutter)
    if rank == 0:
        print("正在执行 CUDA 预热...")
    dummy_input = torch.randint(0, args.vocab_size, (1, 10), device="cuda")
    _ = model(dummy_input, 0)
    torch.cuda.synchronize()
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # 加载权重 | Load weights
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    
    return model, tokenizer, args


def run_interactive(model, tokenizer, args, max_new_tokens, temperature, top_p, repetition_penalty, rank, world_size):
    messages = []
    # 停止 token 示例（可选）| Example stop tokens (optional)
    stop_tokens = [tokenizer.convert_tokens_to_ids("<|eot_id|>")] if hasattr(tokenizer, "convert_tokens_to_ids") else []

    print("\n--- 交互式聊天 (输入 /exit 退出) ---")
    
    while True:
        # 处理分布式输入 | Handle distributed input
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
            print("历史记录已清除。")
            continue
            
        messages.append({"role": "user", "content": prompt})
        # 应用聊天模板 | Apply chat template
        encoded_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        prompt_tokens_list = encoded_prompt.tolist()
        
        print("助手: ", end="", flush=True)
        
        # 流式生成 | Streaming generation
        for next_token_batch in generate_stream(
            model, prompt_tokens_list, max_new_tokens, tokenizer.eos_token_id, 
            stop_tokens, temperature, top_p, repetition_penalty
        ):
            new_token_id = next_token_batch[0] 
            
            # 解码新 token | Decode the new token
            text_chunk = tokenizer.decode([new_token_id], skip_special_tokens=True)
            sys.stdout.write(text_chunk) # 将文本打印到屏幕而不换行 | Print text to screen without newline
            sys.stdout.flush() # 立即刷新输出 | Flush output immediately
            
        print() # 结束时换行 | Newline at the end

        # 重建最终回复以保存到历史记录 | Reconstruct final reply to save to history
        # (在生产环境中，你会累积 ID 并在最后解码) | (In production, you would accumulate IDs and decode at the end)
        messages.append({"role": "assistant", "content": "[已生成]"})


def run_batch(model, tokenizer, args, input_file, max_new_tokens, temperature, top_p, repetition_penalty, rank, world_size):
    if rank == 0:
        print(f"正在从文件读取提示词: {input_file}")
    with open(input_file) as f:
        prompts = [line.strip() for line in f.readlines()]
    
    # 批处理限制 | Batch processing limit
    prompts = prompts[:args.max_batch_size]
    prompt_tokens = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt").tolist()[0] 
        for p in prompts
    ]
    
    print("开始批量生成...")
    # 对于批处理，我们不使用视觉流，只是收集结果 | For batch, we don't use visual streaming, just collect results
    generated_tokens_lists = [[] for _ in prompts]
    
    # 使用 batch 的技巧： | Trick to use generator in batch:
    # 当前的 generate_stream 返回 List[int] (每批一个 token)。| The current generate_stream returns List[int] (one token per batch).
    # 我们需要重新组织。| We need to reorganize.
    
    final_tokens = [[] for _ in prompts]
    
    for next_tokens_batch in generate_stream(
        model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, [], temperature, top_p, repetition_penalty
    ):
        for i, token_id in enumerate(next_tokens_batch):
            final_tokens[i].append(token_id)
            
    completions = tokenizer.batch_decode(final_tokens, skip_special_tokens=True)
    
    if rank == 0:
        for prompt, completion in zip(prompts, completions):
            print(f"提示词: {prompt}")
            print(f"生成内容: {completion}\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7) # 默认更有创意 | Default more creative
    parser.add_argument("--top-p", type=float, default=0.9)        # 新参数 | New parameter
    parser.add_argument("--repetition-penalty", type=float, default=1.0) # 新参数 | New parameter
    args_cli = parser.parse_args()
    
    assert args_cli.input_file or args_cli.interactive, "必须指定 --input-file 或 --interactive"

    # 分布式设置 | Distributed Setup
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        dist.init_process_group("nccl")
        
    global print
    if rank != 0:
        print = lambda *_, **__: None # 禁用非主进程的打印 | Disable printing for non-master processes
        
    torch.cuda.set_device(local_rank)
    
    try:
        model, tokenizer, model_args = load_model_components(
            args_cli.ckpt_path, args_cli.config, rank, world_size
        )

        if args_cli.interactive:
            run_interactive(
                model, tokenizer, model_args, 
                args_cli.max_new_tokens, 
                args_cli.temperature, args_cli.top_p, args_cli.repetition_penalty,
                rank, world_size
            )
        else:
            run_batch(
                model, tokenizer, model_args, 
                args_cli.input_file, 
                args_cli.max_new_tokens, 
                args_cli.temperature, args_cli.top_p, args_cli.repetition_penalty,
                rank, world_size
            )
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
