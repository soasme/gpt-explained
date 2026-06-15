"""Runnable chapter demos for the Python code listings."""

from __future__ import annotations

import math
import random

from micro_bpe import demo as bpe_demo
from microgpt import (
    GPTConfig,
    accumulate_gradients,
    causal_mask,
    cosine_similarity,
    cross_entropy_loss,
    demo as gpt_demo,
    dot,
    embed_sequence,
    feed_forward,
    gelu,
    gpt_forward,
    layer_norm_matrix,
    linear_backward,
    make_feed_forward,
    make_gpt_params,
    make_layer_norm,
    make_matrix,
    make_multi_head_attention,
    make_transformer_block,
    matrix_multiply,
    mean,
    multi_head_attention,
    perplexity,
    random_matrix,
    relu,
    scaled_dot_product_attention,
    self_attention,
    sequence_loss,
    sgd_update_matrix,
    sgd_update_scalar,
    sigmoid,
    sinusoidal_encoding,
    softmax,
    softmax_cross_entropy_grad,
    transformer_block,
    variance,
)


# tag::chapter_02[]
def chapter_02() -> dict[str, object]:
    values = [1.0, 2.0, 3.0]
    logits = [1.0, 2.0, 3.0]
    return {
        "dot": dot([1, 2, 3], [4, 5, 6]),
        "matmul": matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        "softmax": softmax(logits),
        "activations": [relu(-1.0), sigmoid(0.0), round(gelu(1.0), 6)],
        "log": math.log(math.e),
        "mean": mean(values),
        "variance": variance(values),
        "gradient_step": sgd_update_scalar(2.0, 0.5, 0.1),
        "cross_entropy": cross_entropy_loss([0.0, 0.0, 2.0], 2),
        "sinusoidal": sinusoidal_encoding(2, 4),
    }
# end::chapter_02[]


# tag::chapter_03[]
def chapter_03() -> dict[str, object]:
    return bpe_demo()
# end::chapter_03[]


# tag::chapter_04[]
def chapter_04(seed: int = 4) -> dict[str, object]:
    rng = random.Random(seed)
    embedding = random_matrix(100, 8, rng)
    x = embed_sequence(embedding, [1, 2, 3])
    return {
        "sequence_shape": (len(x), len(x[0])),
        "dot": dot(embedding[1], embedding[2]),
        "cosine": cosine_similarity(embedding[1], embedding[2]),
    }
# end::chapter_04[]


# tag::chapter_05[]
def chapter_05() -> dict[str, object]:
    encoding = sinusoidal_encoding(4, 6)
    return {
        "shape": (len(encoding), len(encoding[0])),
        "first": encoding[0],
        "second": encoding[1],
    }
# end::chapter_05[]


# tag::chapter_06[]
def chapter_06(seed: int = 6) -> dict[str, object]:
    rng = random.Random(seed)
    x = random_matrix(4, 8, rng)
    wq = random_matrix(8, 8, rng)
    wk = random_matrix(8, 8, rng)
    wv = random_matrix(8, 8, rng)
    output, weights = self_attention(x, wq, wk, wv)
    return {
        "mask": causal_mask(3),
        "output_shape": (len(output), len(output[0])),
        "weight_rows": [sum(row) for row in weights],
    }
# end::chapter_06[]


# tag::chapter_07[]
def chapter_07(seed: int = 7) -> dict[str, object]:
    rng = random.Random(seed)
    x = random_matrix(4, 8, rng)
    params = make_multi_head_attention(8, 2, rng)
    output, weights = multi_head_attention(x, params)
    return {
        "output_shape": (len(output), len(output[0])),
        "num_heads": len(weights),
    }
# end::chapter_07[]


# tag::chapter_08[]
def chapter_08(seed: int = 8) -> dict[str, object]:
    rng = random.Random(seed)
    x = random_matrix(3, 8, rng)
    params = make_feed_forward(8, rng)
    output = feed_forward(x, params)
    return {"output_shape": (len(output), len(output[0]))}
# end::chapter_08[]


# tag::chapter_09[]
def chapter_09(seed: int = 9) -> dict[str, object]:
    rng = random.Random(seed)
    x = random_matrix(3, 8, rng)
    block = make_transformer_block(8, 2, rng)
    normed = layer_norm_matrix(x, make_layer_norm(8))
    output = transformer_block(x, block)
    return {
        "normed_shape": (len(normed), len(normed[0])),
        "output_shape": (len(output), len(output[0])),
    }
# end::chapter_09[]


# tag::chapter_10[]
def chapter_10(seed: int = 10) -> dict[str, object]:
    config = GPTConfig(vocab_size=50, d_model=16, num_heads=4, num_layers=1, max_seq_len=16)
    params = make_gpt_params(config, random.Random(seed))
    logits = gpt_forward([1, 2, 3], params, config)
    return {
        "logits_len": len(logits),
        "prob_sum": sum(softmax(logits)),
        "top": max(range(len(logits)), key=logits.__getitem__),
    }
# end::chapter_10[]


# tag::chapter_11[]
def chapter_11() -> dict[str, object]:
    logits_list = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.791759469],
        [0.0, -1.0, -1.0, -1.0, -1.405465108],
        [-2.0, -2.0, 2.772588722, 0.0, -2.0],
    ]
    true_ids = [3, 4, 0, 2]
    loss = sequence_loss(logits_list, true_ids)
    return {
        "losses": [cross_entropy_loss(logits, true_id) for logits, true_id in zip(logits_list, true_ids)],
        "mean_loss": loss,
        "perplexity": perplexity(loss),
        "gradient": softmax_cross_entropy_grad(logits_list[0], 3),
    }
# end::chapter_11[]


# tag::chapter_12[]
def chapter_12() -> dict[str, object]:
    delta = [0.1, -0.2]
    weights = [[0.5, -0.25, 0.75], [0.1, 0.2, -0.3]]
    x = [1.0, 2.0, 3.0]
    grad_w, grad_x = linear_backward(delta, weights, x)
    total = accumulate_gradients([grad_w, grad_w])
    updated = [row[:] for row in weights]
    sgd_update_matrix(updated, total, 0.1)
    return {
        "scalar_update": sgd_update_scalar(1.0, 0.25, 0.1),
        "grad_w": grad_w,
        "grad_x": grad_x,
        "updated": updated,
    }
# end::chapter_12[]


# tag::chapter_13[]
def chapter_13() -> dict[str, object]:
    return gpt_demo(seed=13)
# end::chapter_13[]


def run_all() -> dict[str, dict[str, object]]:
    return {
        "chapter_02": chapter_02(),
        "chapter_03": chapter_03(),
        "chapter_04": chapter_04(),
        "chapter_05": chapter_05(),
        "chapter_06": chapter_06(),
        "chapter_07": chapter_07(),
        "chapter_08": chapter_08(),
        "chapter_09": chapter_09(),
        "chapter_10": chapter_10(),
        "chapter_11": chapter_11(),
        "chapter_12": chapter_12(),
        "chapter_13": chapter_13(),
    }


def main() -> None:
    for chapter, result in run_all().items():
        print(f"{chapter}: {result}")


if __name__ == "__main__":
    main()
