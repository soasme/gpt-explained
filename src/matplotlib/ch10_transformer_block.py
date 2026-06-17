import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; MONO='monospace'

def block_diagram():
    fig, ax = plt.subplots(figsize=(6, 8), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 6); ax.set_ylim(0, 8)

    boxes = [
        (1.0, 6.5, 4.0, 0.7, BLUE,      'Layer Norm 1'),
        (1.0, 5.0, 4.0, 1.2, GREEN,     'Multi-Head Attention'),
        (1.0, 3.8, 4.0, 0.7, BLUE,      'Layer Norm 2'),
        (1.0, 2.3, 4.0, 1.2, '#CC8800', 'Feed-Forward Network'),
    ]
    for x, y, w, h, col, lbl in boxes:
        r = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.07', lw=2, ec=col, fc='white')
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, lbl, ha='center', va='center',
                fontsize=11, fontfamily=MONO, color=col, fontweight='bold')

    for (_, y1, _, h1, col, _), (_, y2, _, h2, _, _) in [
        (boxes[0], boxes[1]), (boxes[2], boxes[3])
    ]:
        ax.annotate('', xy=(0.4, y2+h2/2), xytext=(0.4, y1+h1/2),
            arrowprops=dict(arrowstyle='->', color='#999999',
                            connectionstyle='arc3,rad=0.4', lw=1.5))
        ax.text(0.1, (y1+h1/2+y2+h2/2)/2, '+', ha='center', va='center',
                fontsize=14, color='#999999')

    ax.annotate('', xy=(3.0, 7.2), xytext=(3.0, 8.0),
        arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    ax.text(3.0, 7.8, 'x (input)', ha='center', fontsize=10,
            fontfamily=MONO, color=DARK)
    ax.annotate('', xy=(3.0, 1.3), xytext=(3.0, 2.3),
        arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    ax.text(3.0, 1.1, 'x (output)', ha='center', fontsize=10,
            fontfamily=MONO, color=DARK)
    ax.text(3.0, 0.4, 'Transformer Block  (stacked N times)',
            ha='center', fontsize=10, fontfamily=MONO, color=DARK)

    plt.tight_layout()
    plt.savefig('images/ch10_block.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def residual_diagram():
    fig, ax = plt.subplots(figsize=(9, 3), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 9); ax.set_ylim(0, 3)

    ax.text(0.5, 1.5, 'x', ha='center', va='center',
            fontsize=16, fontfamily=MONO, color=DARK, fontweight='bold')
    ax.annotate('', xy=(2.0, 1.5), xytext=(0.9, 1.5),
        arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    r = mpatches.FancyBboxPatch((2.0, 1.0), 2.5, 1.0,
        boxstyle='round,pad=0.07', lw=2, ec=BLUE, fc='white')
    ax.add_patch(r)
    ax.text(3.25, 1.5, 'sublayer(x)', ha='center', va='center',
            fontsize=10, fontfamily=MONO, color=BLUE)
    ax.annotate('', xy=(5.5, 1.5), xytext=(4.5, 1.5),
        arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    c = plt.Circle((5.9, 1.5), 0.35, color=GREEN, fill=False, lw=2.5)
    ax.add_patch(c)
    ax.text(5.9, 1.5, '+', ha='center', va='center',
            fontsize=16, color=GREEN, fontweight='bold')
    ax.annotate('', xy=(5.9, 1.85), xytext=(0.5, 2.5),
        arrowprops=dict(arrowstyle='->', color='#999999',
                        connectionstyle='arc3,rad=-0.25', lw=1.8))
    ax.text(3.0, 2.6, 'skip (x)', ha='center', fontsize=9,
            fontfamily=MONO, color='#999999')
    ax.annotate('', xy=(7.5, 1.5), xytext=(6.25, 1.5),
        arrowprops=dict(arrowstyle='->', color=DARK, lw=2))
    ax.text(8.3, 1.5, 'x + sublayer(x)', ha='center', va='center',
            fontsize=10, fontfamily=MONO, color=DARK)

    plt.tight_layout()
    plt.savefig('images/ch10_residual.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    block_diagram()
    residual_diagram()
    print('ch07 done')
