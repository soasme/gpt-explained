"""Forward-pass building blocks for microGPT.

All model components used in both training and inference live here.
Uses only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


# start snippet types
# start snippet vector_type
Vector = list[float]
# end snippet vector_type
# start snippet matrix_type
Matrix = list[list[float]]
# end snippet matrix_type
# end snippet types


# start snippet matrix
# start snippet make_matrix
def make_matrix(rows: int, cols: int, fill: float = 0.0) -> Matrix:
    return [[fill for _ in range(cols)] for _ in range(rows)]


def shape(matrix: Matrix) -> tuple[int, int]:
    return len(matrix), len(matrix[0]) if matrix else 0
# end snippet make_matrix


# start snippet random_matrix
def random_matrix(
    rows: int,
    cols: int,
    rng: random.Random,
    scale: float | None = None,
) -> Matrix:
    scale = math.sqrt(2.0 / (rows + cols)) if scale is None else scale
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]
# end snippet random_matrix
# end snippet matrix


# start snippet vectors
# start snippet vector_ops
# start snippet add_vectors
def add_vectors(left: Vector, right: Vector) -> Vector:
    return [a + b for a, b in zip(left, right)]
# end snippet add_vectors


# start snippet scale_vector
def scale_vector(vector: Vector, scalar: float) -> Vector:
    return [scalar * value for value in vector]
# end snippet scale_vector


# start snippet vector_norm
def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))
# end snippet vector_norm


# start snippet unit_vector
def unit_vector(vector: Vector) -> Vector:
    return scale_vector(vector, 1.0 / vector_norm(vector))
# end snippet unit_vector
# end snippet vector_ops


# start snippet dot
def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
# end snippet dot


# start snippet cosine_similarity
def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return dot(left, right) / (vector_norm(left) * vector_norm(right))
# end snippet cosine_similarity
# end snippet vectors


# start snippet matrix_ops
# start snippet transpose
def transpose(matrix: Matrix) -> Matrix:
    return [list(col) for col in zip(*matrix)]
# end snippet transpose


# start snippet matrix_add
def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(left, right)]
# end snippet matrix_add


# start snippet matrix_scale
def matrix_scale(matrix: Matrix, scalar: float) -> Matrix:
    return [[scalar * value for value in row] for row in matrix]
# end snippet matrix_scale


# start snippet matrix_multiply
def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    right_t = transpose(right)
    return [[dot(row, col) for col in right_t] for row in left]
# end snippet matrix_multiply


# start snippet hstack
def hstack(matrices: Sequence[Matrix]) -> Matrix:
    return [sum((matrix[row] for matrix in matrices), []) for row in range(len(matrices[0]))]
# end snippet hstack
# end snippet matrix_ops


# start snippet softmax
# start snippet softmax_vector
def softmax(logits: Sequence[float]) -> Vector:
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]
# end snippet softmax_vector


# start snippet softmax_rows
def softmax_rows(matrix: Matrix) -> Matrix:
    return [softmax(row) for row in matrix]
# end snippet softmax_rows
# end snippet softmax


# start snippet activations
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    return max(0.0, x)


def gelu(x: float) -> float:
    scale = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(scale * (x + 0.044715 * x**3)))


def gelu_matrix(matrix: Matrix) -> Matrix:
    return [[gelu(value) for value in row] for row in matrix]
# end snippet activations


# start snippet logarithm
# start snippet natural_log
def natural_log(value: float) -> float:
    return math.log(value)
# end snippet natural_log


# start snippet natural_exp
def natural_exp(value: float) -> float:
    return math.exp(value)
# end snippet natural_exp
# end snippet logarithm


# start snippet statistics
# start snippet mean
def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)
# end snippet mean


# start snippet variance
def variance(values: Sequence[float]) -> float:
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / len(values)
# end snippet variance


# start snippet standard_deviation
def standard_deviation(values: Sequence[float]) -> float:
    return math.sqrt(variance(values))
# end snippet standard_deviation
# end snippet statistics


# start snippet embedding
# start snippet embed_token
def embed_token(embedding: Matrix, token_id: int) -> Vector:
    return list(embedding[token_id])
# end snippet embed_token


# start snippet embed_sequence
def embed_sequence(embedding: Matrix, token_ids: Sequence[int]) -> Matrix:
    return [embed_token(embedding, token_id) for token_id in token_ids]
# end snippet embed_sequence
# end snippet embedding


# start snippet positional_encoding
# start snippet sinusoidal_position
def sinusoidal_position(position: int, d_model: int) -> Vector:
    row = []
    for i in range(d_model):
        k = i // 2
        omega = 1.0 / (10000.0 ** (2.0 * k / d_model))
        row.append(math.sin(position * omega) if i % 2 == 0 else math.cos(position * omega))
    return row
# end snippet sinusoidal_position


# start snippet sinusoidal_encoding
def sinusoidal_encoding(max_length: int, d_model: int) -> Matrix:
    return [sinusoidal_position(position, d_model) for position in range(max_length)]
# end snippet sinusoidal_encoding


# start snippet add_positional_encoding
def add_positional_encoding(tokens: Matrix, positions: Matrix) -> Matrix:
    return matrix_add(tokens, positions[: len(tokens)])
# end snippet add_positional_encoding
# end snippet positional_encoding


# start snippet rope
# start snippet rope_rotate_vector
# start snippet rope_angle
def rope_angle(position: int, pair_index: int, d_model: int) -> float:
    return position / (10000.0 ** (2.0 * pair_index / d_model))
# end snippet rope_angle


# start snippet rotate_pair
def rotate_pair(x: float, y: float, angle: float) -> tuple[float, float]:
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    return x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta
# end snippet rotate_pair


# start snippet apply_rope_to_vector
def apply_rope_to_vector(vector: Vector, position: int) -> Vector:
    if len(vector) % 2 != 0:
        raise ValueError("RoPE requires an even vector dimension")

    rotated: Vector = []
    d_model = len(vector)
    for i in range(0, d_model, 2):
        angle = rope_angle(position, i // 2, d_model)
        x, y = rotate_pair(vector[i], vector[i + 1], angle)
        rotated.extend([x, y])
    return rotated
# end snippet apply_rope_to_vector
# end snippet rope_rotate_vector


# start snippet rope_matrix
def apply_rope(matrix: Matrix) -> Matrix:
    return [
        apply_rope_to_vector(row, position)
        for position, row in enumerate(matrix)
    ]
# end snippet rope_matrix
# end snippet rope


# start snippet attention
# start snippet causal_mask
def causal_mask(size: int) -> Matrix:
    return [[0.0 if j <= i else -1.0e9 for j in range(size)] for i in range(size)]
# end snippet causal_mask


# start snippet sdpa
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
# end snippet sdpa


# start snippet rope_attention
def scaled_dot_product_attention_with_rope(
    query: Matrix,
    key: Matrix,
    value: Matrix,
) -> tuple[Matrix, Matrix]:
    return scaled_dot_product_attention(
        apply_rope(query),
        apply_rope(key),
        value,
    )
# end snippet rope_attention
# end snippet attention


# start snippet mha
# start snippet mha_params
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
# end snippet mha_params


# start snippet mha_forward
def multi_head_attention(x: Matrix, params: MultiHeadAttention) -> tuple[Matrix, list[Matrix]]:
    results = [
        self_attention(x, head.wq, head.wk, head.wv)
        for head in params.heads
    ]
    concatenated = hstack([output for output, _weights in results])
    return matrix_multiply(concatenated, params.wo), [weights for _output, weights in results]
# end snippet mha_forward
# end snippet mha


# start snippet ffn
# start snippet make_feed_forward
@dataclass
class FeedForward:
    w1: Matrix
    b1: Vector
    w2: Matrix
    b2: Vector


def make_feed_forward(d_model: int, rng: random.Random) -> FeedForward:
    d_hidden = 4 * d_model
    return FeedForward(
        w1=random_matrix(d_model, d_hidden, rng),
        b1=[0.0] * d_hidden,
        w2=random_matrix(d_hidden, d_model, rng),
        b2=[0.0] * d_model,
    )
# end snippet make_feed_forward


# start snippet add_bias
def add_bias(matrix: Matrix, bias: Vector) -> Matrix:
    return [[value + bias[j] for j, value in enumerate(row)] for row in matrix]
# end snippet add_bias


# start snippet feed_forward
def feed_forward(x: Matrix, params: FeedForward) -> Matrix:
    hidden = gelu_matrix(add_bias(matrix_multiply(x, params.w1), params.b1))
    return add_bias(matrix_multiply(hidden, params.w2), params.b2)
# end snippet feed_forward
# end snippet ffn


# start snippet layer_norm
# start snippet layer_norm_vec
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
# end snippet layer_norm_vec


# start snippet layer_norm_matrix
def layer_norm_matrix(matrix: Matrix, params: LayerNorm) -> Matrix:
    return [layer_norm_vector(row, params) for row in matrix]
# end snippet layer_norm_matrix
# end snippet layer_norm


# start snippet transformer_block
# start snippet transformer_block_params
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
# end snippet transformer_block_params


# start snippet transformer_block_forward
def transformer_block(x: Matrix, params: TransformerBlock) -> Matrix:
    attn_out, _weights = multi_head_attention(layer_norm_matrix(x, params.ln1), params.attention)
    x1 = matrix_add(x, attn_out)
    ffn_out = feed_forward(layer_norm_matrix(x1, params.ln2), params.feed_forward)
    return matrix_add(x1, ffn_out)


def forward_stack(x: Matrix, blocks: Iterable[TransformerBlock]) -> Matrix:
    for block in blocks:
        x = transformer_block(x, block)
    return x
# end snippet transformer_block_forward
# end snippet transformer_block


# start snippet gpt_config
@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    d_model: int
    num_heads: int
    num_layers: int
    max_seq_len: int
# end snippet gpt_config


# start snippet gpt_params
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
# end snippet gpt_params


# start snippet gpt_forward
def gpt_forward(token_ids: Sequence[int], params: GPTParams, config: GPTConfig) -> Vector:
    if len(token_ids) > config.max_seq_len:
        raise ValueError("token_ids exceeds max_seq_len")

    token_embeddings = embed_sequence(params.wte, token_ids)
    position_embeddings = embed_sequence(params.wpe, range(len(token_ids)))
    x = matrix_add(token_embeddings, position_embeddings)
    x = forward_stack(x, params.blocks)
    h = layer_norm_vector(x[-1], params.ln_f)
    return [dot(h, token_vector) for token_vector in params.wte]
# end snippet gpt_forward


# start snippet parameter_count
def count_parameters(config: GPTConfig) -> int:
    d = config.d_model
    return (
        config.vocab_size * d
        + config.max_seq_len * d
        + config.num_layers * (4 * d * d + 8 * d * d + 4 * d)
        + 2 * d
    )
# end snippet parameter_count
