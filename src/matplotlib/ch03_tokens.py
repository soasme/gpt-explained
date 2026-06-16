import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK = '#222222'
BLUE = '#1177BB'
GREEN = '#228811'
LIGHT_BLUE = '#EAF4FB'
LIGHT_GREEN = '#ECF7EE'
MONO = 'monospace'


def token_id_blocks():
    tokens = [('the', 1), ('cat', 2), ('likes', 3), ('token', 4), ('ization', 5)]

    fig, ax = plt.subplots(figsize=(8.2, 2.4), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('off')

    cell_w = 1.18
    cell_h = 0.72
    gap = 0.2
    ax.set_xlim(-0.25, len(tokens) * (cell_w + gap) - gap + 0.25)
    ax.set_ylim(-0.15, 2.25)

    for i, (word, token_id) in enumerate(tokens):
        x = i * (cell_w + gap)

        word_cell = mpatches.Rectangle(
            (x, 1.05), cell_w, cell_h,
            lw=1.4, ec=BLUE, fc=LIGHT_BLUE,
        )
        id_cell = mpatches.Rectangle(
            (x, 0.33), cell_w, cell_h,
            lw=1.4, ec=GREEN, fc=LIGHT_GREEN,
        )
        ax.add_patch(word_cell)
        ax.add_patch(id_cell)

        ax.text(
            x + cell_w / 2, 1.41, word,
            ha='center', va='center',
            fontsize=14, fontfamily=MONO, color=DARK,
        )
        ax.text(
            x + cell_w / 2, 0.69, str(token_id),
            ha='center', va='center',
            fontsize=14, fontfamily=MONO, color=DARK,
        )

    ax.text(-0.08, 1.41, 'token', ha='right', va='center',
            fontsize=11, fontfamily=MONO, color=BLUE)
    ax.text(-0.08, 0.69, 'ID', ha='right', va='center',
            fontsize=11, fontfamily=MONO, color=GREEN)

    plt.tight_layout()
    plt.savefig('images/ch03_token_id_blocks.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    token_id_blocks()
    print('ch03 tokens done')
