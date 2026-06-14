"""
ch10_training.py
Chapter 10: Training — Backpropagation and gradient descent
Generates: images/ch10_training.png
"""
from manim import *
import numpy as np

BG     = "#FFFFFF"
BLUE   = "#1177BB"
GREEN  = "#228811"
YELLOW = "#CC8800"
ORANGE = "#DD6600"
RED    = "#CC2222"
PURPLE = "#6633AA"
MONO   = "Monospace"


class Ch10TrainingDiagram(Scene):
    """Backprop diagram + loss curve side by side."""
    def construct(self):
        self.camera.background_color = BG

        # ── Left: vertical layer stack with forward/backward arrows ──
        layers = [
            ("Input tokens",   GREEN),
            ("Embedding",      YELLOW),
            ("Transformer\nBlocks (×N)", BLUE),
            ("Vocab Projection", ORANGE),
            ("Loss",           RED),
        ]

        boxes = []
        left_x = -3.8
        y_start = 2.0
        y_step  = 1.1

        for i, (label, color) in enumerate(layers):
            box = RoundedRectangle(corner_radius=0.12, width=2.8, height=0.65,
                                   color=color, fill_opacity=0.18)
            lbl = Text(label, font=MONO, font_size=16, color=color)
            lbl.move_to(box)
            g = VGroup(box, lbl)
            g.move_to(LEFT * 3.8 + UP * (y_start - i * y_step))
            boxes.append(g)

        self.play(*[Create(b) for b in boxes], run_time=1.2)
        self.wait(0.2)

        # Forward arrows (green, going down)
        fwd_arrows = []
        for i in range(len(boxes) - 1):
            arr = Arrow(boxes[i].get_bottom(), boxes[i+1].get_top(),
                        color=GREEN, buff=0.05, stroke_width=2)
            fwd_arrows.append(arr)
        fwd_lbl = Text("forward →", font=MONO, font_size=14, color=GREEN)
        fwd_lbl.next_to(boxes[0], LEFT, buff=0.2).shift(DOWN * 0.2)
        self.play(*[GrowArrow(a) for a in fwd_arrows], FadeIn(fwd_lbl), run_time=0.8)

        # Backward arrows (red, going up, offset to the right of forward)
        bwd_arrows = []
        for i in range(len(boxes) - 1, 0, -1):
            src = boxes[i].get_top() + RIGHT * 0.3
            tgt = boxes[i-1].get_bottom() + RIGHT * 0.3
            arr = Arrow(src, tgt, color=RED, buff=0.05,
                        stroke_width=2, max_tip_length_to_length_ratio=0.15)
            bwd_arrows.append(arr)
        bwd_lbl = Text("← gradients", font=MONO, font_size=14, color=RED)
        bwd_lbl.next_to(boxes[-1], LEFT, buff=0.2).shift(UP * 0.2)
        self.play(*[GrowArrow(a) for a in bwd_arrows], FadeIn(bwd_lbl), run_time=0.8)

        # Update formula
        update_eq = Text("W  ←  W − η · ∇W L",
                         font=MONO, font_size=22, color=YELLOW)
        update_eq.next_to(boxes[-1], DOWN, buff=0.35)
        self.play(Write(update_eq))
        self.wait(0.3)

        # ── Right: loss curve over training steps ──
        steps_data = [0, 100, 1000, 10000, 100000]
        loss_data  = [10.82, 6.91, 4.61, 2.30, 0.92]
        log_steps  = [np.log10(max(s, 1)) for s in steps_data]

        ax = Axes(
            x_range=[0, 5.2, 1],
            y_range=[0, 12, 2],
            x_length=5.0,
            y_length=4.2,
            axis_config={"color": BLUE, "stroke_opacity": 0.6,
                         "include_tip": False},
        )
        ax.move_to(RIGHT * 2.2 + DOWN * 0.3)

        self.play(Create(ax))

        # Plot points and smooth curve
        points = [ax.c2p(x, y) for x, y in zip(log_steps, loss_data)]

        curve = VMobject(color=ORANGE, stroke_width=2.5)
        curve.set_points_smoothly(points)
        self.play(Create(curve), run_time=1.0)

        for pt, (s, l) in zip(points, zip(steps_data, loss_data)):
            dot = Dot(pt, color=YELLOW, radius=0.07)
            self.add(dot)

        # Step labels on x-axis
        x_tick_labels = ["0", "100", "1k", "10k", "100k"]
        for x_val, label in zip(log_steps, x_tick_labels):
            t = Text(label, font=MONO, font_size=12, color=BLUE)
            t.next_to(ax.c2p(x_val, 0), DOWN, buff=0.12)
            self.add(t)

        ax_x_lbl = Text("training steps", font=MONO, font_size=14, color=BLUE)
        ax_x_lbl.next_to(ax, DOWN, buff=0.4)
        ax_y_lbl = Text("loss", font=MONO, font_size=14, color=BLUE)
        ax_y_lbl.rotate(PI / 2)
        ax_y_lbl.next_to(ax, LEFT, buff=0.15)
        self.play(FadeIn(ax_x_lbl), FadeIn(ax_y_lbl))

        self.wait(2.5)
