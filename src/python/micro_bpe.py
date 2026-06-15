"""A tiny byte-pair encoding trainer used by Chapter 3."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


# tag::vocab[]
def make_vocab() -> dict[str, int]:
    return {}


def vocab_extend(vocab: dict[str, int], token: str, token_id: int) -> dict[str, int]:
    return vocab | {token: token_id}


def vocab_lookup(vocab: dict[str, int], token: str) -> int | None:
    return vocab.get(token)
# end::vocab[]


# tag::corpus[]
def word_to_tokens(word: str) -> list[str]:
    return list(word) + ["_"]


def words_to_corpus(words: Sequence[str]) -> list[list[str]]:
    return [word_to_tokens(word) for word in words]
# end::corpus[]


# tag::pairs[]
def adjacent_pairs(tokens: Sequence[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


def count_pairs(corpus: Sequence[Sequence[str]]) -> Counter[tuple[str, str]]:
    return Counter(pair for tokens in corpus for pair in adjacent_pairs(tokens))


def most_frequent(counts: Counter[tuple[str, str]]) -> tuple[str, str]:
    return max(counts, key=counts.get)
# end::pairs[]


# tag::merge[]
def merge_pair(tokens: Sequence[str], left: str, right: str) -> list[str]:
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def merge_corpus(corpus: Sequence[Sequence[str]], left: str, right: str) -> list[list[str]]:
    return [merge_pair(tokens, left, right) for tokens in corpus]
# end::merge[]


# tag::train[]
def unique_tokens(corpus: Sequence[Sequence[str]]) -> list[str]:
    return sorted({token for tokens in corpus for token in tokens})


def train_bpe(words: Sequence[str], target_size: int) -> tuple[dict[str, int], list[tuple[str, str]]]:
    corpus = words_to_corpus(words)
    vocab = {token: token_id for token_id, token in enumerate(unique_tokens(corpus))}
    merges = []

    while len(vocab) < target_size:
        counts = count_pairs(corpus)
        if not counts:
            break
        left, right = most_frequent(counts)
        new_token = left + right
        if new_token not in vocab:
            vocab[new_token] = len(vocab)
        merges.append((left, right))
        corpus = merge_corpus(corpus, left, right)

    return vocab, merges
# end::train[]


# tag::encode[]
def encode(word: str, merges: Sequence[tuple[str, str]], vocab: dict[str, int]) -> list[int]:
    tokens = word_to_tokens(word)
    for left, right in merges:
        tokens = merge_pair(tokens, left, right)
    missing = [token for token in tokens if token not in vocab]
    if missing:
        raise ValueError(f"unknown token(s): {missing}")
    return [vocab[token] for token in tokens]
# end::encode[]


# tag::demo[]
def demo() -> dict[str, object]:
    words = ["low", "lower", "newest", "widest"]
    vocab, merges = train_bpe(words, 20)
    return {
        "vocab": vocab,
        "merges": merges,
        "low": encode("low", merges, vocab),
        "lower": encode("lower", merges, vocab),
    }
# end::demo[]


def main() -> None:
    result = demo()
    vocab = result["vocab"]
    assert isinstance(vocab, dict)
    print("Vocabulary:")
    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        print(f"  {token_id} -> {token}")
    print()
    print('Encoding "low":  ', result["low"])
    print('Encoding "lower":', result["lower"])


if __name__ == "__main__":
    main()
