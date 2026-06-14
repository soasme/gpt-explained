"""
ch03_positional_encoding.py
Chapter 3: Sinusoidal PE heatmap + waveforms
Generates: images/ch03-positional-encoding.png
"""
from manim import *
import numpy as np

BG     = "#FFFFFF"
BLUE   = "#1177BB"
GREEN  = "#228811"
YELLOW = "#CC8800"
ORANGE = "#DD6600"
MONO   = "Monospace"


class Ch03PositionalEncoding(Scene):
    """Heatmap of the PE matrix [T × d]."""

    def construct(self):
        self.camera.background_color = BG

        T, d = 20, 16
        base = 10000.0
        PE = np.zeros((T, d))
        for t in range(T):
            for i in range(d):
                k = i // 2
                omega = 1.0 / (base ** (2 * k / d))
                PE[t, i] = np.sin(t * omega) if i % 2 == 0 else np.cos(t * omega)

        # Draw as a grid of colored squares
        cell_w = 0.38
        cell_h = 0.30
        grid = VGroup()
        for t in range(T):
            for i in range(d):
                val = PE[t, i]
                # Map val ∈ [-1,1] to color
                r = max(0.0, val)
                b = max(0.0, -val)
                color = rgb_to_color([r * 0.9 + 0.1, 0.3, b * 0.9 + 0.1])
                sq = Square(side_length=min(cell_w, cell_h),
                            fill_color=color, fill_opacity=0.85,
                            stroke_width=0.3, stroke_color=BG)
                sq.move_to(RIGHT * i * cell_w + DOWN * t * cell_h)
                grid.add(sq)
        grid.center().shift(DOWN * 0.4)

        self.play(Create(grid), run_time=2.5)
        self.wait(0.5)

        # Axis labels
        dim_label = Text("dimension →", font=MONO, font_size=16, color=BLUE)
        dim_label.next_to(grid, DOWN, buff=0.25)
        pos_label = Text("position ↓", font=MONO, font_size=16, color=BLUE)
        pos_label.rotate(PI / 2)
        pos_label.next_to(grid, LEFT, buff=0.25)

        self.play(FadeIn(dim_label), FadeIn(pos_label))
        self.wait(0.5)

        # Annotate: high freq column vs low freq column
        high_box = SurroundingRectangle(
            VGroup(*[grid[t * d] for t in range(T)]),
            color=YELLOW, stroke_width=2, buff=0.05)
        low_box = SurroundingRectangle(
            VGroup(*[grid[t * d + d - 2] for t in range(T)]),
            color=GREEN, stroke_width=2, buff=0.05)
        high_lbl = Text("high freq\n(dim 0)", font=MONO, font_size=14, color=YELLOW)
        high_lbl.next_to(high_box, DOWN, buff=0.1)
        low_lbl = Text("low freq\n(dim 14)", font=MONO, font_size=14, color=GREEN)
        low_lbl.next_to(low_box, DOWN, buff=0.1)

        self.play(Create(high_box), Create(low_box))
        self.play(FadeIn(high_lbl), FadeIn(low_lbl))
        self.wait(2.5)

class Ch03SinWaves(Scene):
    """Show 3 sine waves at different frequencies."""
    def construct(self):
        self.camera.background_color = BG

        freqs = [1.0, 0.1, 0.01]
        colors = [YELLOW, GREEN, ORANGE]
        labels = ["dim 0 (ω=1.0)", "dim 4 (ω=0.1)", "dim 8 (ω=0.01)"]
        y_offsets = [1.5, 0.0, -1.5]

        axes = []
        for i, (freq, col, lbl, yo) in enumerate(zip(freqs, colors, labels, y_offsets)):
            ax = Axes(x_range=[0, 20, 5], y_range=[-1.2, 1.2, 0.5],
                      x_length=8, y_length=1.2,
                      axis_config={"color": col, "stroke_opacity": 0.4},
                      tips=False)
            ax.shift(DOWN * yo - UP * 0.2)
            curve = ax.plot(lambda t, f=freq: np.sin(t * f),
                            x_range=[0, 20], color=col, stroke_width=2.5)
            label = Text(lbl, font=MONO, font_size=16, color=col)
            label.next_to(ax, LEFT, buff=0.2)
            self.play(Create(ax), Create(curve), FadeIn(label), run_time=0.8)

        self.wait(2.0)
