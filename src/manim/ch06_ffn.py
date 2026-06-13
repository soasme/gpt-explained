"""
ch06_ffn.py
Chapter 6: Feed-Forward Network expand→GELU→contract
Generates: images/ch06-ffn.png
"""
from manim import *
import numpy as np

BG     = "#1C1C1C"
BLUE   = "#58C4DD"
GREEN  = "#83C167"
YELLOW = "#FFFF00"
ORANGE = "#FF9000"
MONO   = "Monospace"


class Ch06FFN(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Feed-Forward Network (FFN)", font=MONO,
                     font_size=34, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # --- Layer diagram ---
        def make_layer_box(label, dims, color, pos):
            box = RoundedRectangle(corner_radius=0.15,
                                   width=3.0, height=0.8,
                                   color=color, fill_opacity=0.18)
            lbl = Text(f"{label}\n{dims}", font=MONO, font_size=18, color=color)
            lbl.move_to(box)
            g = VGroup(box, lbl)
            g.move_to(pos)
            return g

        layers = [
            make_layer_box("Input", "[T × d]",    GREEN,  UP * 2.0),
            make_layer_box("Linear W₁", "[T × 4d]", YELLOW, UP * 0.6),
            make_layer_box("GELU",  "[T × 4d]",   ORANGE, DOWN * 0.8),
            make_layer_box("Linear W₂", "[T × d]",  BLUE,   DOWN * 2.2),
        ]

        self.play(*[Create(l) for l in layers])
        self.wait(0.3)

        # Arrows
        for i in range(len(layers) - 1):
            arr = Arrow(layers[i].get_bottom(), layers[i+1].get_top(),
                        color=WHITE, buff=0.05, stroke_width=2)
            self.play(GrowArrow(arr), run_time=0.4)

        self.wait(0.5)

        # --- GELU curve inset ---
        ax = Axes(x_range=[-3, 3, 1], y_range=[-0.5, 3, 0.5],
                  x_length=4.0, y_length=2.5,
                  axis_config={"color": BLUE, "stroke_opacity": 0.5},
                  tips=False)
        ax.to_edge(RIGHT, buff=0.5).shift(DOWN * 0.3)

        def gelu(x):
            return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

        gelu_curve = ax.plot(gelu, x_range=[-3, 3], color=ORANGE, stroke_width=2.5)
        relu_curve = ax.plot(lambda x: max(0, x), x_range=[-3, 3],
                             color=GREEN, stroke_width=1.5,
                             stroke_dasharray=[8, 4])

        gelu_lbl = Text("GELU", font=MONO, font_size=16, color=ORANGE)
        gelu_lbl.next_to(ax, UP, buff=0.1).shift(RIGHT * 0.5)
        relu_lbl = Text("ReLU (dashed)", font=MONO, font_size=14, color=GREEN)
        relu_lbl.next_to(gelu_lbl, DOWN, buff=0.1)

        self.play(Create(ax), Create(gelu_curve), Create(relu_curve),
                  FadeIn(gelu_lbl), FadeIn(relu_lbl))
        self.wait(1.5)

        formula = MathTex(r"\text{FFN}(x) = \text{GELU}(xW_1 + b_1)\,W_2 + b_2",
                          font_size=26, color=YELLOW)
        formula.to_edge(DOWN, buff=0.45)
        self.play(Write(formula))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))


class Ch06MemoryInterpretation(Scene):
    """FFN as key-value memory."""
    def construct(self):
        self.camera.background_color = BG
        title = Text("FFN as Associative Memory", font=MONO,
                     font_size=30, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # 3 neuron rows
        examples = [
            ("input: France…", "→ neuron 42 fires", "→ output: Paris"),
            ("input: was born", "→ neuron 107 fires", "→ output: in…"),
            ("input: [MASK]",  "→ neuron 55 fires",  "→ output: is a…"),
        ]

        for i, (inp, mid, out) in enumerate(examples):
            y = 1.2 - i * 1.2
            in_lbl  = Text(inp, font=MONO, font_size=18, color=GREEN)
            mid_lbl = Text(mid, font=MONO, font_size=18, color=YELLOW)
            out_lbl = Text(out, font=MONO, font_size=18, color=ORANGE)
            row = VGroup(in_lbl, mid_lbl, out_lbl)
            row.arrange(RIGHT, buff=0.5)
            row.move_to(UP * y)
            self.play(FadeIn(row), run_time=0.6)

        note = Text("Neurons in W₁ are 'keys'; columns of W₂ are 'values'",
                    font=MONO, font_size=16, color=BLUE)
        note.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))
