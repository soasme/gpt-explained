"""
Chapter 9: Loss — Cross-Entropy & Perplexity
Scenes:
  Ch09LossScene      → ch09_loss.png
"""
from manim import *

BG      = "#1C1C1C"
BLUE    = "#58C4DD"
GREEN   = "#83C167"
YELLOW  = "#FFFF00"
RED     = "#FF6B6B"
GREY    = "#888888"

config.background_color = BG


class Ch09LossScene(Scene):
    def construct(self):
        title = Text("Cross-Entropy Loss", font_size=36, color=BLUE).to_edge(UP)
        self.play(Write(title))

        # ── Left panel: probability bar chart ──
        labels   = ["the","cat","sat","on","mat"]
        probs    = [0.05, 0.10, 0.60, 0.20, 0.05]
        true_idx = 3                              # "on" is the true next token

        bar_w, bar_gap = 0.55, 0.15
        max_h          = 2.5
        total_w        = len(labels) * (bar_w + bar_gap)
        start_x        = -3.8

        bars  = VGroup()
        ticks = VGroup()
        for i, (lab, p) in enumerate(zip(labels, probs)):
            x   = start_x + i * (bar_w + bar_gap)
            h   = p * max_h / max(probs)
            col = YELLOW if i == true_idx else BLUE
            bar = Rectangle(width=bar_w, height=h, fill_color=col,
                            fill_opacity=0.85, stroke_width=0)
            bar.move_to([x + bar_w / 2, -1.2 + h / 2, 0])
            prob_txt = Text(f"{p:.2f}", font_size=16, color=WHITE)
            prob_txt.next_to(bar, UP, buff=0.06)
            word_txt = Text(lab, font_size=16, color=WHITE)
            word_txt.next_to(bar, DOWN, buff=0.06)
            bars.add(bar, prob_txt, word_txt)

        # True-token arrow
        true_bar_x = start_x + true_idx * (bar_w + bar_gap) + bar_w / 2
        arrow = Arrow(
            [true_bar_x, 0.6, 0],
            [true_bar_x, 0.1, 0],
            color=YELLOW, buff=0.05, stroke_width=3,
        )
        arrow_lbl = Text("true next token", font_size=16, color=YELLOW)
        arrow_lbl.next_to(arrow, UP, buff=0.06)

        self.play(Create(bars), GrowArrow(arrow), Write(arrow_lbl))

        # ── Right panel: loss formula ──
        eq1 = MathTex(
            r"\mathcal{L} = -\log P(\text{true})",
            font_size=34, color=WHITE,
        ).move_to([2.5, 0.8, 0])

        eq2 = MathTex(
            r"= -\log(0.20) \approx 1.61",
            font_size=30, color=GREEN,
        ).next_to(eq1, DOWN, buff=0.35)

        ppl = MathTex(
            r"\text{PPL} = e^{\mathcal{L}} = e^{1.61} \approx 5.0",
            font_size=28, color=YELLOW,
        ).next_to(eq2, DOWN, buff=0.35)

        self.play(Write(eq1))
        self.play(Write(eq2))
        self.play(Write(ppl))

        # ── Loss curve strip at bottom ──
        ax = Axes(
            x_range=[0, 1.05, 0.25],
            y_range=[0, 5.5,  1.0],
            x_length=3.8,
            y_length=2.0,
            axis_config={"color": GREY, "include_tip": False},
            x_axis_config={"numbers_to_include": [0.25, 0.5, 0.75, 1.0]},
            y_axis_config={"numbers_to_include": [1, 2, 3, 4, 5]},
        ).move_to([2.6, -1.3, 0])

        neg_log_curve = ax.plot(
            lambda p: -np.log(max(p, 1e-6)),
            x_range=[0.02, 1.0],
            color=RED,
            stroke_width=2.5,
        )
        ax_lbl = Text("-log(p)", font_size=16, color=RED).next_to(ax, UP, buff=0.08)

        # Dot at p=0.20
        dot = Dot(ax.c2p(0.20, -np.log(0.20)), color=YELLOW, radius=0.08)
        dot_lbl = Text("p=0.20\nL≈1.61", font_size=13, color=YELLOW)
        dot_lbl.next_to(dot, RIGHT, buff=0.1)

        self.play(Create(ax), Create(neg_log_curve), Write(ax_lbl))
        self.play(FadeIn(dot), Write(dot_lbl))

        self.wait(2)
