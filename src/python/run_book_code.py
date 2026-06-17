"""End-to-end checks for every runnable code chapter."""

from __future__ import annotations

import math

from chapter_demos import run_all


def close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return abs(left - right) <= tolerance


def assert_probability_rows(rows: list[float], tolerance: float = 1.0e-9) -> None:
    for total in rows:
        assert close(total, 1.0, tolerance), total


def main() -> None:
    results = run_all()

    ch02 = results["chapter_02"]
    assert ch02["dot"] == 32
    assert ch02["matmul"] == [[19, 22], [43, 50]]
    assert close(sum(ch02["softmax"]), 1.0)
    assert ch02["activations"] == [0.0, 0.5, 0.841192]
    assert close(ch02["log"], 1.0)
    assert close(ch02["mean"], 2.0)
    assert close(ch02["variance"], 2.0 / 3.0)
    assert close(ch02["gradient_step"], 1.95)

    ch03 = results["chapter_03"]
    assert ch03["low"] == [16]
    assert ch03["lower"] == [19]

    assert results["chapter_04"]["sequence_shape"] == (3, 8)
    assert results["chapter_05"]["shape"] == (4, 6)
    assert results["chapter_06"]["output_shape"] == (4, 8)
    assert_probability_rows(results["chapter_06"]["weight_rows"], tolerance=1.0e-8)

    rope = results["chapter_07"]
    assert rope["rotated_shape"] == (4, 8)
    assert rope["output_shape"] == (4, 8)
    assert_probability_rows(rope["weight_rows"], tolerance=1.0e-8)

    assert results["chapter_08"]["output_shape"] == (4, 8)
    assert results["chapter_08"]["num_heads"] == 2
    assert results["chapter_09"]["output_shape"] == (3, 8)
    assert results["chapter_10"]["normed_shape"] == (3, 8)
    assert results["chapter_10"]["output_shape"] == (3, 8)

    ch11 = results["chapter_11"]
    assert ch11["logits_len"] == 50
    assert close(ch11["prob_sum"], 1.0, tolerance=1.0e-8)
    assert isinstance(ch11["top"], int)

    ch12 = results["chapter_12"]
    assert len(ch12["losses"]) == 4
    assert close(ch12["mean_loss"], sum(ch12["losses"]) / 4)
    assert close(ch12["perplexity"], math.exp(ch12["mean_loss"]))
    assert close(sum(ch12["gradient"]), 0.0, tolerance=1.0e-8)

    ch13 = results["chapter_13"]
    assert close(ch13["scalar_update"], 0.975)
    assert len(ch13["grad_w"]) == 2
    assert len(ch13["grad_w"][0]) == 3
    assert len(ch13["grad_x"]) == 3

    ch14 = results["chapter_14"]
    assert ch14["parameters"] == 30144
    assert len(ch14["logits_first_10"]) == 10
    assert len(ch14["generated"]) == 10
    assert len(ch14["full_sequence"]) == 13
    assert all(0 <= token < 100 for token in ch14["generated"])

    print("All chapter code checks passed.")


if __name__ == "__main__":
    main()
