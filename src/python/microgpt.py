"""Small, runnable GPT building blocks for GPT Explained.

The book uses this file as the source of truth for its code listings.
It intentionally uses only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


Vector = list[float]
Matrix = list[list[float]]


# tag::matrix[]
def make_matrix(rows: int, cols: int, fill: float = 0.0) -> Matrix:
    return [[fill for _ in range(cols)] for _ in range(rows)]


def shape(matrix: Matrix) -> tuple[int, int]:
    return len(matrix), len(matrix[0]) if matrix else 0


def random_matrix(
    rows: int,
    cols: int,
    rng: random.Random,
    scale: float | None = None,
) -> Matrix:
    scale = math.sqrt(2.0 / (rows + cols)) if scale is None else scale
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]
# end::matrix[]


# tag::vectors[]
def add_vectors(left: Vector, right: Vector) -> Vector:
    return [a + b for a, b in zip(left, right)]


def scale_vector(vector: Vector, scalar: float) -> Vector:
    return [scalar * value for value in vector]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return dot(left, right) / (vector_norm(left) * vector_norm(right))
# end::vectors[]


# tag::matrix_ops[]
def transpose(matrix: Matrix) -> Matrix:
    return [list(col) for col in zip(*matrix)]


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(left, right)]


def matrix_scale(matrix: Matrix, scalar: float) -> Matrix:
    return [[scalar * value for value in row] for row in matrix]


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    right_t = transpose(right)
    return [[dot(row, col) for col in right_t] for row in left]


def hstack(matrices: Sequence[Matrix]) -> Matrix:
    return [sum((matrix[row] for matrix in matrices), []) for row in range(len(matrices[0]))]
# end::matrix_ops[]


# tag::softmax[]
def softmax(logits: Sequence[float]) -> Vector:
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def softmax_rows(matrix: Matrix) -> Matrix:
    return [softmax(row) for row in matrix]
# end::softmax[]


# tag::activations[]
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    return max(0.0, x)


def gelu(x: float) -> float:
    scale = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(scale * (x + 0.044715 * x**3)))


def gelu_matrix(matrix: Matrix) -> Matrix:
    return [[gelu(value) for value in row] for row in matrix]
# end::activations[]


# tag::logarithm[]
def natural_log(value: float) -> float:
    return math.log(value)
# end::logarithm[]


# tag::statistics[]
def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)
# end::statistics[]


# tag::embedding[]
def embed_token(embedding: Matrix, token_id: int) -> Vector:
    return list(embedding[token_id])


def embed_sequence(embedding: Matrix, token_ids: Sequence[int]) -> Matrix:
    return [embed_token(embedding, token_id) for token_id in token_ids]
# end::embedding[]


# tag::positional_encoding[]
def sinusoidal_position(position: int, d_model: int) -> Vector:
    row = []
    for i in range(d_model):
        k = i // 2
        omega = 1.0 / (10000.0 ** (2.0 * k / d_model))
        row.append(math.sin(position * omega) if i % 2 == 0 else math.cos(position * omega))
    return row


def sinusoidal_encoding(max_length: int, d_model: int) -> Matrix:
    return [sinusoidal_position(position, d_model) for position in range(max_length)]


def add_positional_encoding(tokens: Matrix, positions: Matrix) -> Matrix:
    return matrix_add(tokens, positions[: len(tokens)])
# end::positional_encoding[]


# tag::attention[]
def causal_mask(size: int) -> Matrix:
    return [[0.0 if j <= i else -1.0e9 for j in range(size)] for i in range(size)]


def scaled_dot_product_attention(query: Matrix, key: Matrix, value: Matrix) -> tuple[Matrix, Matrix]:
    d_key = shape(query)[1]
    scores = matrix_scale(matrix_multiply(query, transpose(key)), 1.0 / math.sqrt(d_key))
    masked_scores = matrix_add(scores, causal_mask(len(scores)))
    weights = softmax_rows(masked_scores)
    return matrix_multiply(weights, value), weights


def self_attention(x: Matrix, wq: Matrix, wk: Matrix, wv: Matrix) -> tuple[Matrix, Matrix]:
    return scaled_dot_product_attention(
        matrix_multiply(x, wq),
        matrix_multiply(x, wk),
        matrix_multiply(x, wv),
    )
# end::attention[]


# tag::mha[]
@dataclass
class AttentionHead:
    wq: Matrix
    wk: Matrix
    wv: Matrix


@dataclass
class MultiHeadAttention:
    heads: list[AttentionHead]
    wo: Matrix


def make_multi_head_attention(d_model: int, num_heads: int, rng: random.Random) -> MultiHeadAttention:
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    d_key = d_model // num_heads
    heads = [
        AttentionHead(
            random_matrix(d_model, d_key, rng),
            random_matrix(d_model, d_key, rng),
            random_matrix(d_model, d_key, rng),
        )
        for _ in range(num_heads)
    ]
    return MultiHeadAttention(heads=heads, wo=random_matrix(d_model, d_model, rng))


def multi_head_attention(x: Matrix, params: MultiHeadAttention) -> tuple[Matrix, list[Matrix]]:
    results = [
        self_attention(x, head.wq, head.wk, head.wv)
        for head in params.heads
    ]
    concatenated = hstack([output for output, _weights in results])
    return matrix_multiply(concatenated, params.wo), [weights for _output, weights in results]
# end::mha[]


# tag::ffn[]
@dataclass
class FeedForward:
    w1: Matrix
    b1: Vector
    w2: Matrix
    b2: Vector


def add_bias(matrix: Matrix, bias: Vector) -> Matrix:
    return [[value + bias[j] for j, value in enumerate(row)] for row in matrix]


def make_feed_forward(d_model: int, rng: random.Random) -> FeedForward:
    d_hidden = 4 * d_model
    return FeedForward(
        w1=random_matrix(d_model, d_hidden, rng),
        b1=[0.0] * d_hidden,
        w2=random_matrix(d_hidden, d_model, rng),
        b2=[0.0] * d_model,
    )


def feed_forward(x: Matrix, params: FeedForward) -> Matrix:
    hidden = gelu_matrix(add_bias(matrix_multiply(x, params.w1), params.b1))
    return add_bias(matrix_multiply(hidden, params.w2), params.b2)
# end::ffn[]


# tag::layer_norm[]
@dataclass
class LayerNorm:
    gamma: Vector
    beta: Vector
    epsilon: float = 1.0e-5


def make_layer_norm(d_model: int) -> LayerNorm:
    return LayerNorm(gamma=[1.0] * d_model, beta=[0.0] * d_model)


def layer_norm_vector(vector: Vector, params: LayerNorm) -> Vector:
    avg = mean(vector)
    std = math.sqrt(variance(vector) + params.epsilon)
    return [
        params.gamma[i] * ((value - avg) / std) + params.beta[i]
        for i, value in enumerate(vector)
    ]


def layer_norm_matrix(matrix: Matrix, params: LayerNorm) -> Matrix:
    return [layer_norm_vector(row, params) for row in matrix]
# end::layer_norm[]


# tag::transformer_block[]
@dataclass
class TransformerBlock:
    attention: MultiHeadAttention
    feed_forward: FeedForward
    ln1: LayerNorm
    ln2: LayerNorm


def make_transformer_block(d_model: int, num_heads: int, rng: random.Random) -> TransformerBlock:
    return TransformerBlock(
        attention=make_multi_head_attention(d_model, num_heads, rng),
        feed_forward=make_feed_forward(d_model, rng),
        ln1=make_layer_norm(d_model),
        ln2=make_layer_norm(d_model),
    )


def transformer_block(x: Matrix, params: TransformerBlock) -> Matrix:
    attn_out, _weights = multi_head_attention(layer_norm_matrix(x, params.ln1), params.attention)
    x1 = matrix_add(x, attn_out)
    ffn_out = feed_forward(layer_norm_matrix(x1, params.ln2), params.feed_forward)
    return matrix_add(x1, ffn_out)


def forward_stack(x: Matrix, blocks: Iterable[TransformerBlock]) -> Matrix:
    for block in blocks:
        x = transformer_block(x, block)
    return x
# end::transformer_block[]


# tag::gpt_config[]
@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    max_seq_len: int
# end::gpt_config[]


# tag::gpt_params[]
@dataclass
class GPTParams:
    wte: Matrix
    wpe: Matrix
    blocks: list[TransformerBlock]
    ln_f: LayerNorm


def make_gpt_params(config: GPTConfig, rng: random.Random) -> GPTParams:
    return GPTParams(
        wte=random_matrix(config.vocab_size, config.d_model, rng),
        wpe=random_matrix(config.max_seq_len, config.d_model, rng),
        blocks=[
            make_transformer_block(config.d_model, config.num_heads, rng)
            for _ in range(config.num_layers)
        ],
        ln_f=make_layer_norm(config.d_model),
    )
# end::gpt_params[]


# tag::gpt_forward[]
def gpt_forward(token_ids: Sequence[int], params: GPTParams, config: GPTConfig) -> Vector:
    if len(token_ids) > config.max_seq_len:
        raise ValueError("token_ids exceeds max_seq_len")

    token_embeddings = embed_sequence(params.wte, token_ids)
    position_embeddings = embed_sequence(params.wpe, range(len(token_ids)))
    x = matrix_add(token_embeddings, position_embeddings)
    x = forward_stack(x, params.blocks)
    h = layer_norm_vector(x[-1], params.ln_f)
    return [dot(h, token_vector) for token_vector in params.wte]
# end::gpt_forward[]


# tag::generation[]
def temperature_scale(logits: Sequence[float], temperature: float) -> Vector:
    return [value / temperature for value in logits]


def top_k_filter(logits: Sequence[float], k: int) -> Vector:
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


# tag::loss[]
def cross_entropy_loss(logits: Sequence[float], true_id: int) -> float:
    probabilities = softmax(logits)
    return -math.log(max(probabilities[true_id], 1.0e-12))


def sequence_loss(logits_list: Sequence[Sequence[float]], true_ids: Sequence[int]) -> float:
    losses = [cross_entropy_loss(logits, true_id) for logits, true_id in zip(logits_list, true_ids)]
    return sum(losses) / len(losses)


def perplexity(loss: float) -> float:
    return math.exp(loss)


def softmax_cross_entropy_grad(logits: Sequence[float], true_id: int) -> Vector:
    grad = softmax(logits)
    grad[true_id] -= 1.0
    return grad
# end::loss[]


# tag::training[]
def sgd_update_scalar(weight: float, gradient: float, learning_rate: float) -> float:
    return weight - learning_rate * gradient


def linear_backward(delta: Vector, weights: Matrix, x: Vector) -> tuple[Matrix, Vector]:
    grad_w = [[delta_i * x_j for x_j in x] for delta_i in delta]
    grad_x = [sum(weights[i][j] * delta[i] for i in range(len(delta))) for j in range(len(x))]
    return grad_w, grad_x


def accumulate_gradients(gradients: Sequence[Matrix]) -> Matrix:
    rows, cols = shape(gradients[0])
    total = make_matrix(rows, cols)
    for gradient in gradients:
        total = matrix_add(total, gradient)
    return total


def sgd_update_matrix(weights: Matrix, gradient: Matrix, learning_rate: float) -> None:
    for i, row in enumerate(weights):
        for j, value in enumerate(row):
            row[j] = value - learning_rate * gradient[i][j]
# end::training[]


# tag::parameter_count[]
def count_parameters(config: GPTConfig) -> int:
    d = config.d_model
    return (
        config.vocab_size * d
        + config.max_seq_len * d
        + config.num_layers * (4 * d * d + 8 * d * d + 4 * d)
        + 2 * d
    )
# end::parameter_count[]


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


def main() -> None:
    result = demo()
    config = result["config"]
    assert isinstance(config, GPTConfig)
    print("microGPT in Python")
    print(f"vocab-size  : {config.vocab_size}")
    print(f"d-model     : {config.d_model}")
    print(f"num-heads   : {config.num_heads}")
    print(f"num-layers  : {config.num_layers}")
    print(f"max-seq-len : {config.max_seq_len}")
    print(f"parameters  : {result['parameters']}")
    print("Logits (first 10):", [round(value, 4) for value in result["logits_first_10"]])
    print(f"Top predicted token: {result['top_token']} (prob {result['top_probability']:.4f})")
    print("Generated:", result["generated"])
    print("Full sequence:", result["full_sequence"])


if __name__ == "__main__":
    main()
