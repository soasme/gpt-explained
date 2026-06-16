#!/usr/bin/env python3
"""Render every Matplotlib-generated book image in one Python process."""

from pathlib import Path

from ch02_activations import activations_plot
from ch02_dot_product import dot_product
from ch02_embeddings import embedding_lookup, semantic_space
from ch02_gradient import gradient_plot
from ch02_matrix_multiply import mat_multiply
from ch02_softmax import softmax_plot
from ch02_vectors import vectors
from ch03_tokens import token_id_blocks
from ch03_positional_encoding import pe_matrix, sin_waves
from ch04_attention import attn_weights, qkv_diagram
from ch05_multi_head_attention import mha
from ch06_ffn import ffn_diagram, ffn_memory
from ch07_transformer_block import block_diagram, residual_diagram
from ch08_vocab_projection import generation_loop, vocab_projection
from ch09_loss import loss_diagram
from ch10_training import training_diagram
from ch13_rope import rope_plot


RENDER_STEPS = [
    ("ch02_activations", [activations_plot]),
    ("ch02_dot_product", [dot_product]),
    ("ch02_embeddings", [embedding_lookup, semantic_space]),
    ("ch02_gradient", [gradient_plot]),
    ("ch02_matrix_multiply", [mat_multiply]),
    ("ch02_softmax", [softmax_plot]),
    ("ch02_vectors", [vectors]),
    ("ch03_tokens", [token_id_blocks]),
    ("ch03_positional_encoding", [pe_matrix, sin_waves]),
    ("ch04_attention", [qkv_diagram, attn_weights]),
    ("ch05_multi_head_attention", [mha]),
    ("ch06_ffn", [ffn_diagram, ffn_memory]),
    ("ch07_transformer_block", [block_diagram, residual_diagram]),
    ("ch08_vocab_projection", [vocab_projection, generation_loop]),
    ("ch09_loss", [loss_diagram]),
    ("ch10_training", [training_diagram]),
    ("ch13_rope", [rope_plot]),
]


def main():
    for name, functions in RENDER_STEPS:
        print(f"Running {name} ...")
        for render in functions:
            render()

    png_count = len(list(Path("images").glob("*.png")))
    print()
    print("Done! PNGs in images/:")
    print(f"{png_count:8d}")


if __name__ == "__main__":
    main()
