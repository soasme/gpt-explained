"""
ch08_vocab_projection.py
Chapter 8: Vocabulary Projection — logits → probabilities
Generates: images/ch08-vocab-projection.png
"""
from manim import *
import numpy as np

BG     = "#1C1C1C"
BLUE   = "#58C4DD"
GREEN  = "#83C167"
YELLOW = "#FFFF00"
ORANGE = "#FF9000"
PURPLE = "#AA88FF"
MONO   = "Monospace"


def softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


class Ch08VocabProjection(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Vocabulary Projection: Vector → Probabilities",
                     font=MONO, font_size=28, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Step 1: hidden state vector
        h_vals = [0.3, -0.1, 0.8, 0.2]
        h_box = VGroup(*[
            VGroup(
                Square(side_length=0.55, color=GREEN, fill_opacity=0.15),
                Text(f"{v:.1f}", font=MONO, font_size=18, color=GREEN)
            ) for v in h_vals
        ])
        h_box.arrange(RIGHT, buff=0.06)
        h_box.move_to(UP * 2.0 + LEFT * 3.0)
        h_lbl = Text("h ∈ ℝ⁴", font=MONO, font_size=18, color=GREEN)
        h_lbl.next_to(h_box, UP, buff=0.15)
        self.play(Create(h_box), FadeIn(h_lbl))
        self.wait(0.3)

        # Arrow → unembedding
        arr1 = Arrow(h_box.get_right(), h_box.get_right() + RIGHT * 1.5,
                     color=YELLOW, buff=0.1, stroke_width=2)
        wu_lbl = Text("× Wᵤ\n(= Eᵀ)", font=MONO, font_size=16, color=YELLOW)
        wu_lbl.next_to(arr1, UP, buff=0.1)
        self.play(GrowArrow(arr1), FadeIn(wu_lbl))

        # Logits bar chart
        logits = np.array([0.21, -0.31, -0.18, 0.51, -0.33])
        vocab = ["cat", "dog", "mat", "sat", "ran"]
        n_vocab = len(logits)

        ax = Axes(x_range=[-0.5, n_vocab - 0.5, 1],
                  y_range=[-0.6, 0.8, 0.3],
                  x_length=4.0, y_length=2.5,
                  axis_config={"color": BLUE, "stroke_opacity": 0.5,
                               "include_numbers": False},
                  tips=False)
        ax.shift(RIGHT * 1.8 + UP * 0.5)

        bars = VGroup()
        for i, (logit, word) in enumerate(zip(logits, vocab)):
            height = logit
            color = YELLOW if i == np.argmax(logits) else BLUE
            bar = Rectangle(
                width=0.5,
                height=abs(height),
                color=color, fill_opacity=0.7, stroke_width=1)
            bar.move_to(ax.c2p(i, height / 2))
            w_lbl = Text(word, font=MONO, font_size=13, color=color)
            w_lbl.next_to(ax.c2p(i, -0.6), DOWN, buff=0.05)
            val_lbl = Text(f"{logit:.2f}", font=MONO, font_size=12, color=WHITE)
            val_lbl.next_to(bar, UP if height > 0 else DOWN, buff=0.05)
            bars.add(VGroup(bar, w_lbl, val_lbl))

        logits_title = Text("logits", font=MONO, font_size=16, color=YELLOW)
        logits_title.next_to(ax, UP, buff=0.1)

        self.play(Create(ax), FadeIn(logits_title))
        self.play(*[Create(b) for b in bars], run_time=1.0)
        self.wait(0.5)

        # Arrow → softmax → probs
        arr2 = Arrow(ax.get_right(), ax.get_right() + RIGHT * 1.2,
                     color=ORANGE, buff=0.1, stroke_width=2)
        softmax_lbl = Text("softmax", font=MONO, font_size=16, color=ORANGE)
        softmax_lbl.next_to(arr2, UP, buff=0.1)

        probs = softmax(logits)
        prob_bars = VGroup()
        ax2 = Axes(x_range=[-0.5, n_vocab - 0.5, 1],
                   y_range=[0, 0.5, 0.1],
                   x_length=3.5, y_length=2.5,
                   axis_config={"color": BLUE, "stroke_opacity": 0.5,
                                "include_numbers": False},
                   tips=False)
        ax2.next_to(arr2, RIGHT, buff=0.3).shift(DOWN * 0.3)

        for i, (prob, word) in enumerate(zip(probs, vocab)):
            color = ORANGE if i == np.argmax(probs) else GREEN
            bar = Rectangle(width=0.45, height=prob,
                            color=color, fill_opacity=0.7, stroke_width=1)
            bar.move_to(ax2.c2p(i, prob / 2))
            w_lbl = Text(word, font=MONO, font_size=12, color=color)
            w_lbl.next_to(ax2.c2p(i, 0), DOWN, buff=0.05)
            pct_lbl = Text(f"{prob:.0%}", font=MONO, font_size=11, color=WHITE)
            pct_lbl.next_to(bar, UP, buff=0.04)
            prob_bars.add(VGroup(bar, w_lbl, pct_lbl))

        probs_title = Text("P(next token)", font=MONO, font_size=16, color=ORANGE)
        probs_title.next_to(ax2, UP, buff=0.1)

        self.play(GrowArrow(arr2), FadeIn(softmax_lbl))
        self.play(Create(ax2), FadeIn(probs_title))
        self.play(*[Create(b) for b in prob_bars], run_time=1.0)
        self.wait(0.5)

        # Highlight winner
        winner = Text(f'Most likely: "{vocab[np.argmax(probs)]}"  ({max(probs):.1%})',
                      font=MONO, font_size=18, color=ORANGE)
        winner.to_edge(DOWN, buff=0.5)
        self.play(Write(winner))
        self.wait(3.0)
        self.play(FadeOut(Group(*self.mobjects)))


class Ch08GenerationLoop(Scene):
    """Autoregressive generation loop diagram."""
    def construct(self):
        self.camera.background_color = BG
        title = Text("Autoregressive Generation", font=MONO,
                     font_size=32, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        steps = [
            ("Prompt", '"the cat sat on"', GREEN),
            ("Forward pass", "GPT(tokens) → logits", YELLOW),
            ("Sample", "token = sample(softmax(logits))", ORANGE),
            ("Append", '"the cat sat on [new]"', BLUE),
            ("Repeat", "until max length or stop token", PURPLE),
        ]

        prev = None
        for i, (step_lbl, detail, col) in enumerate(steps):
            y = 1.8 - i * 1.0
            box = RoundedRectangle(corner_radius=0.1, width=5.5, height=0.6,
                                    color=col, fill_opacity=0.15)
            box.move_to(UP * y)
            lbl = Text(f"{i+1}. {step_lbl}: {detail}", font=MONO,
                       font_size=16, color=col)
            lbl.move_to(box)
            g = VGroup(box, lbl)
            self.play(Create(g), run_time=0.5)
            if prev is not None:
                arr = Arrow(prev.get_bottom(), g.get_top(),
                            color=WHITE, buff=0.04, stroke_width=1.8)
                self.play(GrowArrow(arr), run_time=0.3)
            prev = g

        # Loop back arrow
        loop_arr = CurvedArrow(
            prev.get_right(), steps[1][0] and
            VGroup(*self.mobjects).get_right() + UP * 2.5 + LEFT * 0.2,
            angle=TAU / 5, color=YELLOW, stroke_width=2)

        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))
