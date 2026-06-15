"""Token generation and end-to-end demo for microGPT.

Imports from common and the Python standard library.
"""

from __future__ import annotations

import random
from typing import Sequence

from common import (
    GPTConfig,
    GPTParams,
    count_parameters,
    gpt_forward,
    make_gpt_params,
    softmax,
)


# tag::generation[]
def temperature_scale(logits: Sequence[float], temperature: float) -> list[float]:
    return [value / temperature for value in logits]


def top_k_filter(logits: Sequence[float], k: int) -> list[float]:
    keep = {index for index, _value in sorted(enumerate(logits), key=lambda item: item[1], reverse=True)[:k]}
    return [value if index in keep else -1.0e9 for index, value in enumerate(logits)]


def sample_token(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def generate(
    params: GPTParams,
    config: GPTConfig,
    start_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    rng: random.Random,
) -> list[int]:
    tokens = list(start_ids)
    generated = []
    for _ in range(max_new_tokens):
        logits = gpt_forward(tokens, params, config)
        probabilities = softmax(top_k_filter(temperature_scale(logits, temperature), top_k))
        next_token = sample_token(probabilities, rng)
        tokens.append(next_token)
        generated.append(next_token)
    return generated
# end::generation[]


# tag::demo[]
def demo(seed: int = 7) -> dict[str, object]:
    config = GPTConfig(vocab_size=100, d_model=32, num_heads=4, num_layers=2, max_seq_len=64)
    model_rng = random.Random(seed)
    sample_rng = random.Random(seed + 1)
    params = make_gpt_params(config, model_rng)
    logits = gpt_forward([1, 2, 3, 4, 5], params, config)
    probabilities = softmax(logits)
    top_token = max(range(len(probabilities)), key=probabilities.__getitem__)
    generated = generate(params, config, [1, 2, 3], 10, temperature=0.8, top_k=10, rng=sample_rng)
    return {
        "config": config,
        "parameters": count_parameters(config),
        "logits_first_10": logits[:10],
        "top_token": top_token,
        "top_probability": probabilities[top_token],
        "generated": generated,
        "full_sequence": [1, 2, 3] + generated,
    }
# end::demo[]


if __name__ == "__main__":
    result = demo()
    print(f"Parameters: {result['parameters']:,}")
    print(f"Top token: {result['top_token']} (p={result['top_probability']:.4f})")
    print(f"Generated: {result['generated']}")
