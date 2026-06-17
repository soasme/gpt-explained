import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'
YELLOW='#CC8800'; RED='#CC2222'; MONO='monospace'

def vocab_projection():
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 12); ax.set_ylim(0, 4)

    stages = [
        (0.3, 1.4, 1.4, 1.2, BLUE,   'h\n[d]'),
        (2.5, 0.5, 2.0, 3.0, GREEN,  'W_unembed\n[d x |V|]'),
        (5.5, 0.8, 1.2, 2.4, YELLOW, 'logits\n[|V|]'),
        (7.5, 1.2, 1.6, 1.6, RED,    'softmax'),
        (9.8, 0.8, 1.8, 2.4, BLUE,   'P(next\ntoken)'),
    ]
    prev_right = None
    for x, y, w, h, col, lbl in stages:
        r = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.07', lw=2, ec=col, fc='white')
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, lbl, ha='center', va='center',
                fontsize=10, fontfamily=MONO, color=col)
        if prev_right:
            ax.annotate('', xy=(x, y+h/2), xytext=(prev_right, y+h/2),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.8))
        prev_right = x+w

    plt.tight_layout()
    plt.savefig('images/ch11_projection.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

def generation_loop():
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 8); ax.set_ylim(0, 5)

    nodes = [
        (4.0, 4.0, 'context tokens', BLUE),
        (6.5, 2.5, 'transformer\nforward pass', GREEN),
        (4.0, 1.0, 'sample next\ntoken', YELLOW),
        (1.5, 2.5, 'append to\ncontext', RED),
    ]
    positions = [(x, y) for x, y, _, _ in nodes]
    for x, y, lbl, col in nodes:
        r = mpatches.FancyBboxPatch((x-1.1, y-0.4), 2.2, 0.8,
            boxstyle='round,pad=0.07', lw=2, ec=col, fc='white')
        ax.add_patch(r)
        ax.text(x, y, lbl, ha='center', va='center',
                fontsize=9, fontfamily=MONO, color=col)

    for (x1,y1), (x2,y2) in [
        (positions[0], positions[1]),
        (positions[1], positions[2]),
        (positions[2], positions[3]),
        (positions[3], positions[0]),
    ]:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=1.8,
                            connectionstyle='arc3,rad=0.15'))

    ax.text(4.0, 0.3, 'Repeat until end-of-sequence token',
            ha='center', fontsize=9, fontfamily=MONO, color=DARK)

    plt.tight_layout()
    plt.savefig('images/ch11_generation.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    vocab_projection()
    generation_loop()
    print('ch08 done')
