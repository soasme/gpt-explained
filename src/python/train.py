"""Loss functions and gradient-descent optimiser for microGPT.

Imports only from common and the Python standard library.
"""

from __future__ import annotations

import math
from typing import Sequence

from common import Matrix, Vector, make_matrix, matrix_add, shape, softmax


# tag::loss[]
# tag::cross_entropy_loss[]
def cross_entropy_loss(logits: Sequence[float], true_id: int) -> float:
    probabilities = softmax(logits)
    return -math.log(max(probabilities[true_id], 1.0e-12))
# end::cross_entropy_loss[]


# tag::sequence_loss[]
def sequence_loss(logits_list: Sequence[Sequence[float]], true_ids: Sequence[int]) -> float:
    losses = [cross_entropy_loss(logits, true_id) for logits, true_id in zip(logits_list, true_ids)]
    return sum(losses) / len(losses)


# tag::perplexity[]
def perplexity(loss: float) -> float:
    return math.exp(loss)
# end::perplexity[]
# end::sequence_loss[]


# tag::loss_grad[]
def softmax_cross_entropy_grad(logits: Sequence[float], true_id: int) -> Vector:
    grad = softmax(logits)
    grad[true_id] -= 1.0
    return grad
# end::loss_grad[]
# end::loss[]


# tag::training[]
# tag::sgd_update_scalar[]
def sgd_update_scalar(weight: float, gradient: float, learning_rate: float) -> float:
    return weight - learning_rate * gradient
# end::sgd_update_scalar[]


# tag::linear_backward[]
def linear_backward(delta: Vector, weights: Matrix, x: Vector) -> tuple[Matrix, Vector]:
    grad_w = [[delta_i * x_j for x_j in x] for delta_i in delta]
    grad_x = [sum(weights[i][j] * delta[i] for i in range(len(delta))) for j in range(len(x))]
    return grad_w, grad_x
# end::linear_backward[]


# tag::accumulate_gradients[]
def accumulate_gradients(gradients: Sequence[Matrix]) -> Matrix:
    rows, cols = shape(gradients[0])
    total = make_matrix(rows, cols)
    for gradient in gradients:
        total = matrix_add(total, gradient)
    return total
# end::accumulate_gradients[]


# tag::sgd_update_matrix[]
def sgd_update_matrix(weights: Matrix, gradient: Matrix, learning_rate: float) -> None:
    for i, row in enumerate(weights):
        for j, value in enumerate(row):
            row[j] = value - learning_rate * gradient[i][j]
# end::sgd_update_matrix[]
# end::training[]
