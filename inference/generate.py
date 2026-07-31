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


def _next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample (temperature > 0) or argmax (temperature == 0) a token from logits."""
    return sample(logits, temperature) if temperature > 0 else logits.argmax(dim=-1)


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

    When MTP modules are available (``model.mtp_modules`` is non-empty), uses
    the DeepSeek-V3 Multi-Token Prediction block as a single-token draft model
    for speculative decoding:

    1. ``forward_with_hidden`` produces the next token T_N plus the hidden
       state h_{N-1} that produced it (one main-model seqlen=1 forward).
    2. ``forward_mtp`` drafts a speculation T_{N+1}^spec from (T_N, h_{N-1})
       using a single extra transformer block — cheap relative to a main
       forward. Write T_{N+1}^spec to slot ``cur_pos + 1`` and mark a
       speculation pending.
    3. The next iteration's verify step runs a seqlen=1 ``forward_with_hidden``
       over the accepted token at slot ``cur_pos`` (= T_N), recomputing the
       verifier logits for slot ``cur_pos + 1``. The verified token T_{N+1}
       is compared against the speculation: on a match the speculated token is
       accepted as-is; on a miss it is replaced in place by the verified token.
    4. The verify step's hidden is then used to draft the next speculation
       (step 2 again), closing the loop.

    With a single MTP module the scheme is correctness-preserving: the output
    sequence is determined entirely by the main model, with MTP speculation
    only ever overwritten on a miss. There is no wall-clock speedup with a
    pure seqlen=1 verify because each accepted token still costs one main
    forward — the speedup surface is the seqlen=2 verify (process the accepted
    token and the speculation in a single forward, comparing per-position
    logits), which the cache-aware causal mask in ``forward_with_hidden`` now
    supports and which is left as a follow-up extension. In the current shape
    the contribution is the working MTP draft+verify plumbing plus the mask
    fix that unblocks multi-token mid-decode forwards.

    The scheme degrades to standard autoregressive decoding when MTP modules
    are absent (``num_nextn_predict_layers == 0``), so configs that do not ship
    MTP weights are unaffected.

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
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    # Source the device from the model parameters so the loop works on CPU for
    # testing as well as on the production CUDA device.
    device = next(model.parameters()).device
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=device)
    prompt_mask = tokens != -1
    mtp_enabled = len(model.mtp_modules) > 0

    spec_valid = False

    cur_pos = min(prompt_lens)
    while cur_pos < total_len:
        if spec_valid:
            # ---- Verification step (seqlen=1) ----
            # Invariant: slots 0..cur_pos-1 hold accepted tokens whose KV cache
            # is populated.  tokens[:, cur_pos] holds the speculation to
            # verify.  Re-forward the last accepted token (slot cur_pos-1) at
            # start_pos = cur_pos-1 to reproduce the main-model prediction for
            # slot cur_pos and compare against the speculation. This seqlen=1
            # forward also re-populates KV slot cur_pos-1 (idempotent, the
            # token is unchanged) and produces a fresh hidden at slot
            # cur_pos-1 used to re-speculate on a miss / speculate next on a
            # hit.
            verify_input = tokens[:, cur_pos - 1:cur_pos]
            assert 0 <= cur_pos - 1 < total_len, \
                f"verify position {cur_pos - 1} out of bounds (total_len={total_len})"
            verify_hidden, verify_logit = model.forward_with_hidden(
                verify_input, cur_pos - 1,
            )
            verified_token = _next_token(verify_logit, temperature).squeeze()
            tokens[:, cur_pos] = torch.where(
                prompt_mask[:, cur_pos], tokens[:, cur_pos], verified_token,
            )
            finished |= torch.logical_and(
                ~prompt_mask[:, cur_pos], verified_token == eos_id,
            )
            spec_valid = False
            if finished.all():
                break
            # Slot cur_pos is settled; advance.
            prev_pos = cur_pos
            cur_pos = cur_pos + 1
            # Speculate slot cur_pos using the hidden at slot cur_pos-1 (the
            # forward we just ran) and the chosen token at cur_pos-1.
            if mtp_enabled and cur_pos < total_len:
                last_token = tokens[:, prev_pos - 1]
                mtp_logits = model.forward_mtp(0, last_token, verify_hidden, prev_pos - 1)
                speculative_token = _next_token(mtp_logits, temperature)
                speculative_token = torch.where(
                    prompt_mask[:, cur_pos], tokens[:, cur_pos], speculative_token,
                )
                tokens[:, cur_pos] = speculative_token.squeeze()
                spec_valid = True
            continue

        # ---- Standard seqlen=1 step (first token, or after a forced rewind) ----
        sf_input = tokens[:, prev_pos:cur_pos]
        assert 0 <= prev_pos < cur_pos <= total_len, \
            f"seqlen=1 range [{prev_pos},{cur_pos}) invalid (total_len={total_len})"
        hidden, logits = model.forward_with_hidden(sf_input, prev_pos)
        first_token = _next_token(logits, temperature)
        first_token = torch.where(
            tokens[:, cur_pos] != -1, tokens[:, cur_pos], first_token
        )
        tokens[:, cur_pos] = first_token.squeeze()
        finished |= torch.logical_and(
            ~prompt_mask[:, cur_pos], first_token == eos_id
        )
        if finished.all():
            break
        spec_valid = False
        if mtp_enabled and cur_pos + 1 < total_len:
            mtp_logits = model.forward_mtp(0, first_token, hidden, cur_pos)
            speculative_token = _next_token(mtp_logits, temperature)
            speculative_token = torch.where(
                prompt_mask[:, cur_pos + 1], tokens[:, cur_pos + 1], speculative_token,
            )
            tokens[:, cur_pos + 1] = speculative_token.squeeze()
            spec_valid = True
        prev_pos = cur_pos
        cur_pos = cur_pos + 1

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
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
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
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
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
