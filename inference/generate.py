import os
import json
from argparse import ArgumentParser
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    eos_id: int


class TokenSampler:
    @staticmethod
    def sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Samples a token from the logits using temperature scaling.

        Args:
            logits (torch.Tensor): The logits tensor for token predictions.
            temperature (float): Temperature for scaling logits.

        Returns:
            torch.Tensor: The sampled token.
        """
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


class TextGenerator:
    def __init__(self, model: Transformer, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        config: GenerationConfig
    ) -> List[List[int]]:
        """
        Generates new tokens based on the given prompt tokens.

        Args:
            prompt_tokens: A list of lists containing the prompt tokens for each sequence.
            config: Generation configuration parameters.

        Returns:
            List[List[int]]: Generated tokens for each sequence.
        """
        prompt_lens = [len(t) for t in prompt_tokens]
        assert max(prompt_lens) <= self.model.max_seq_len
        
        total_len = min(self.model.max_seq_len, config.max_new_tokens + max(prompt_lens))
        tokens = self._initialize_tokens(prompt_tokens, total_len)
        
        completion_tokens = self._generate_tokens(
            tokens, prompt_lens, total_len, config
        )
        return completion_tokens

    def _initialize_tokens(
        self, prompt_tokens: List[List[int]], total_len: int
    ) -> torch.Tensor:
        tokens = torch.full(
            (len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda"
        )
        for i, t in enumerate(prompt_tokens):
            tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        return tokens

    def _generate_tokens(
        self,
        tokens: torch.Tensor,
        prompt_lens: List[int],
        total_len: int,
        config: GenerationConfig
    ) -> List[List[int]]:
        prev_pos = 0
        finished = torch.tensor([False] * len(prompt_lens), device="cuda")
        prompt_mask = tokens != -1

        for cur_pos in range(min(prompt_lens), total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = self._get_next_token(logits, config.temperature)
            next_token = torch.where(
                prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            
            tokens[:, cur_pos] = next_token
            finished |= torch.logical_and(
                ~prompt_mask[:, cur_pos], next_token == config.eos_id
            )
            prev_pos = cur_pos
            
            if finished.all():
                break

        return self._process_completion_tokens(
            tokens, prompt_lens, config.max_new_tokens, config.eos_id
        )

    def _get_next_token(
        self, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        if temperature > 0:
            return TokenSampler.sample(logits, temperature)
        return logits.argmax(dim=-1)

    def _process_completion_tokens(
        self,
        tokens: torch.Tensor,
        prompt_lens: List[int],
        max_new_tokens: int,
        eos_id: int
    ) -> List[List[int]]:
        completion_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_lens[i]:prompt_lens[i] + max_new_tokens]
            if eos_id in toks:
                toks = toks[:toks.index(eos_id)]
            completion_tokens.append(toks)
        return completion_tokens


class DistributedEnvironment:
    def __init__(self):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    def setup(self):
        if self.world_size > 1:
            dist.init_process_group("nccl")
        if self.rank != 0:
            global print
            print = lambda *_, **__: None
        torch.cuda.set_device(self.local_rank)

    def cleanup(self):
        if self.world_size > 1:
            dist.destroy_process_group()

    def broadcast_prompt(self, prompt: Optional[str] = None) -> str:
        if self.world_size == 1:
            return input(">>> ")
        elif self.rank == 0:
            prompt = input(">>> ")
            objects = [prompt]
            dist.broadcast_object_list(objects, 0)
            return prompt
        else:
            objects = [None]
            dist.broadcast_object_list(objects, 0)
            return objects[0]


class ChatSession:
    def __init__(
        self,
        generator: TextGenerator,
        config: GenerationConfig,
        dist_env: DistributedEnvironment
    ):
        self.generator = generator
        self.config = config
        self.dist_env = dist_env
        self.messages = []

    def run_interactive(self):
        while True:
            prompt = self.dist_env.broadcast_prompt()
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                self.messages.clear()
                continue

            completion = self._process_message(prompt)
            print(completion)
            self.messages.append({"role": "assistant", "content": completion})

    def run_batch(self, input_file: str):
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= self.generator.model.args.max_batch_size

        completions = self._process_batch(prompts)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    def _process_message(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        prompt_tokens = self.generator.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True
        )
        completion_tokens = self.generator.generate(
            [prompt_tokens], self.config
        )
        return self.generator.tokenizer.decode(
            completion_tokens[0], skip_special_tokens=True
        )

    def _process_batch(self, prompts: List[str]) -> List[str]:
        prompt_tokens = [
            self.generator.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True
            )
            for prompt in prompts
        ]
        completion_tokens = self.generator.generate(
            prompt_tokens, self.config
        )
        return self.generator.tokenizer.batch_decode(
            completion_tokens, skip_special_tokens=True
        )


def initialize_model(
    ckpt_path: str, config_path: str, dist_env: DistributedEnvironment
) -> Tuple[Transformer, Any]:
    """Initialize the model and tokenizer."""
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    print(args)

    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # Warmup
    tokenizer.decode(
        TextGenerator(model, tokenizer).generate(
            [tokenizer.encode("DeepSeek")],
            GenerationConfig(max_new_tokens=2, temperature=1.0, eos_id=-1)
        )[0]
    )

    load_model(
        model,
        os.path.join(
            ckpt_path,
            f"model{dist_env.rank}-mp{dist_env.world_size}.safetensors"
        )
    )
    return model, tokenizer


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    dist_env = DistributedEnvironment()
    dist_env.setup()

    model, tokenizer = initialize_model(ckpt_path, config, dist_env)
    generator = TextGenerator(model, tokenizer)
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_id=tokenizer.eos_token_id
    )

    session = ChatSession(generator, gen_config, dist_env)
    
    if interactive:
        session.run_interactive()
    else:
        session.run_batch(input_file)

    dist_env.cleanup()


if __name__ == "__main__":
    parser = ArgumentParser(description="Distributed text generation system")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    
    assert args.input_file or args.interactive
    main(
        args.ckpt_path,
        args.config,
        args.input_file,
        args.interactive,
        args.max_new_tokens,
        args.temperature
    )