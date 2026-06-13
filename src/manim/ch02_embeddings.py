"""
ch02_embeddings.py
Chapter 2: Embedding Matrix Lookup Diagram
Generates: images/ch02-embedding-lookup.png
"""
from manim import *

BG     = "#1C1C1C"
BLUE   = "#58C4DD"
GREEN  = "#83C167"
YELLOW = "#FFFF00"
ORANGE = "#FF9000"
MONO   = "Monospace"


class Ch02EmbeddingLookup(Scene):
    def construct(self):
        self.camera.background_color = BG

        title = Text("Embedding Matrix Lookup", font=MONO, font_size=34,
                     color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title), run_time=1.0)
        self.wait(0.3)

        # ---- Draw embedding matrix (|V|=6, d=4) ----
        V, D = 6, 4
        cell_w, cell_h = 0.65, 0.5

        def make_matrix_mob(V, D, highlighted_rows=()):
            """Create a VGroup representing a matrix grid."""
            cells = VGroup()
            for i in range(V):
                for j in range(D):
                    color = YELLOW if i in highlighted_rows else BLUE
                    opacity = 0.5 if i in highlighted_rows else 0.1
                    sq = Rectangle(width=cell_w, height=cell_h,
                                   color=color, fill_opacity=opacity,
                                   stroke_width=1.5)
                    sq.move_to(RIGHT * j * cell_w + DOWN * i * cell_h)
                    cells.add(sq)
            return cells

        matrix = make_matrix_mob(V, D)
        matrix.move_to(LEFT * 3.5 + UP * 0.3)

        # Row labels
        row_labels = VGroup()
        for i in range(V):
            lbl = Text(f"tok {i}", font=MONO, font_size=16, color=WHITE)
            lbl.next_to(matrix[i * D], LEFT, buff=0.2)
            row_labels.add(lbl)

        # Column header
        col_label = Text(f"E  [{V}×{D}]", font=MONO, font_size=18, color=BLUE)
        col_label.next_to(matrix, UP, buff=0.25)

        self.play(Create(matrix), FadeIn(row_labels), FadeIn(col_label))
        self.wait(0.8)

        # ---- Input: token IDs ----
        ids = [2, 4, 0]
        id_text = Text("Input IDs: " + str(ids), font=MONO, font_size=22,
                       color=GREEN)
        id_text.to_edge(DOWN, buff=1.8)
        self.play(Write(id_text))
        self.wait(0.5)

        # ---- Highlight selected rows and draw arrows ----
        output_rows = VGroup()
        colors = [ORANGE, GREEN, YELLOW]

        for k, (tok_id, col) in enumerate(zip(ids, colors)):
            # Highlight the row
            row_cells = VGroup(*[matrix[tok_id * D + j] for j in range(D)])
            self.play(row_cells.animate.set_fill(col, opacity=0.6).set_color(col),
                      run_time=0.5)

            # Extract vector label
            row_x = matrix[tok_id * D].get_left()[0]
            row_y = matrix[tok_id * D].get_center()[1]
            src = np.array([row_x + cell_w * D, row_y, 0])
            tgt = np.array([1.5, 1.2 - k * 1.0, 0])
            arrow = Arrow(src, tgt, color=col, buff=0.05, stroke_width=2)

            vec_label = Text(f"x[{k}] = E[{tok_id}]", font=MONO,
                             font_size=20, color=col)
            vec_label.move_to(tgt + RIGHT * 1.3)

            # Small vector illustration
            vec_rect = Rectangle(width=2.0, height=0.42, color=col,
                                 fill_opacity=0.2, stroke_width=1.5)
            vec_rect.next_to(vec_label, RIGHT, buff=0.25)
            vec_inner = Text("[ 0.12  -0.83  0.41  0.67 ]", font=MONO,
                             font_size=11, color=col)
            vec_inner.move_to(vec_rect)

            output_rows.add(VGroup(arrow, vec_label, vec_rect, vec_inner))
            self.play(GrowArrow(arrow), FadeIn(vec_label),
                      Create(vec_rect), FadeIn(vec_inner))
            self.wait(0.6)

        self.wait(1.0)

        # ---- Show output matrix shape ----
        shape_note = Text("Output X: shape [3 × 4]", font=MONO,
                          font_size=22, color=YELLOW)
        shape_note.next_to(id_text, UP, buff=0.3)
        self.play(Write(shape_note))
        self.wait(2.0)
        self.play(FadeOut(Group(*self.mobjects)))


class Ch02SemanticSpace(Scene):
    """King–Man+Woman≈Queen in 2D."""
    def construct(self):
        self.camera.background_color = BG

        title = Text("Semantic Arithmetic in Embedding Space",
                     font=MONO, font_size=28, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        ax = Axes(x_range=[-3.5, 3.5, 1], y_range=[-3, 3, 1],
                  x_length=7, y_length=5,
                  axis_config={"color": BLUE, "stroke_opacity": 0.5},
                  tips=False)
        ax.move_to(DOWN * 0.3)
        self.play(Create(ax))

        # Approximate 2D projections
        points = {
            "king":   np.array([ 2.0,  1.5]),
            "queen":  np.array([ 2.0, -0.5]),
            "man":    np.array([-1.5,  1.5]),
            "woman":  np.array([-1.5, -0.5]),
        }
        colors = {"king": YELLOW, "queen": ORANGE, "man": GREEN, "woman": BLUE}
        dots = {}

        for word, pt in points.items():
            d = Dot(ax.c2p(*pt), color=colors[word], radius=0.12)
            lbl = Text(word, font=MONO, font_size=20, color=colors[word])
            lbl.next_to(d, UP, buff=0.12)
            dots[word] = d
            self.play(Create(d), FadeIn(lbl), run_time=0.5)

        self.wait(0.5)

        # Draw the "king - man + woman" arrow
        arr = Arrow(ax.c2p(*points["king"]), ax.c2p(*points["queen"]),
                    color=YELLOW, buff=0.0, stroke_width=2.5)
        equation = MathTex(r"\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}",
                           font_size=28, color=YELLOW)
        equation.to_edge(DOWN, buff=0.5)
        self.play(GrowArrow(arr))
        self.play(Write(equation))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))
