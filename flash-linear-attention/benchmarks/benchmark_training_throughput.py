# -*- coding: utf-8 -*-

import argparse
import time

import torch
from torch.cuda import max_memory_allocated, memory_allocated
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoModelForCausalLM

from fla.models.abc import ABCConfig
from fla.models.delta_net import DeltaNetConfig
from fla.models.gla import GLAConfig
from fla.models.linear_attn import LinearAttentionConfig
from fla.models.retnet import RetNetConfig

configs = {
    'abc': ABCConfig,
    'delta_net': DeltaNetConfig,
    'gla': GLAConfig,
    'linear_attn': LinearAttentionConfig,
    'retnet': RetNetConfig,
}


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def profile_throughput(name: str, batch_size: int = 8, seq_len: int = 2048, warmup_steps: int = 16, steps: int = 32):
    device = torch.device('cuda')
    config = configs[name]()
    model = AutoModelForCausalLM.from_config(config).cuda().to(torch.bfloat16)
    print(f"Initializing {name} model from the config")
    print(config)
    print(model)
    print(f"Allocated memory after initialization: {sizeof_fmt(memory_allocated(device))}")

    optimizer = AdamW(model.parameters())

    bar = trange(warmup_steps)
    for _ in bar:
        # forward pass
        tokens = torch.randint(high=config.vocab_size, size=(batch_size, seq_len)).cuda()
        outputs = model(tokens, labels=tokens)
        # backward pass
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        bar.set_description_str(f"Max memory allocated: {sizeof_fmt(max_memory_allocated(device))}")

    torch.cuda.synchronize(device)
    start = time.time()
    bar = trange(steps)
    total_tokens = 0
    for _ in bar:
        # forward pass
        tokens = torch.randint(high=config.vocab_size, size=(batch_size, seq_len), device=device)
        outputs = model(tokens, labels=tokens)
        # backward pass
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch_size * seq_len
        duration = time.time() - start
        bar.set_description_str(f"Thoughput: {total_tokens / duration:8.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='retnet')
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seq_len", default=2048, type=int)
    parser.add_argument("--warmup_steps", default=16, type=int)
    parser.add_argument("--steps", default=32, type=int)
    args = parser.parse_args()
    profile_throughput(args.name, args.batch_size, args.seq_len, args.warmup_steps, args.steps)
