import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'
YELLOW='#CC8800'; RED='#CC2222'; MONO='monospace'

def training_diagram():
    fig, ax = plt.subplots(figsize=(6, 9), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 6); ax.set_ylim(0, 9)

    boxes = [
        (1.0, 7.5, 4.0, 0.8, BLUE,   'Training data (tokens)'),
        (1.0, 5.8, 4.0, 0.8, GREEN,  'Forward pass (predict)'),
        (1.0, 4.1, 4.0, 0.8, YELLOW, 'Loss  L = -log P(true)'),
        (1.0, 2.4, 4.0, 0.8, RED,    'Backprop (chain rule)'),
        (1.0, 0.7, 4.0, 0.8, BLUE,   'Weight update: theta -= alpha * grad'),
    ]
    for x, y, w, h, col, lbl in boxes:
        r = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.07', lw=2, ec=col, fc='white')
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, lbl, ha='center', va='center',
                fontsize=9, fontfamily=MONO, color=col)

    ys = [b[1] for b in boxes]
    for i in range(len(ys)-1):
        ax.annotate('', xy=(3.0, ys[i+1]+0.8), xytext=(3.0, ys[i]),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=2))

    ax.annotate('', xy=(1.0, 6.2), xytext=(0.5, 1.1),
        arrowprops=dict(arrowstyle='->', color='#999999',
                        connectionstyle='arc3,rad=0.3', lw=1.5,
                        linestyle='dashed'))
    ax.text(0.1, 3.5, 'repeat', fontsize=9, fontfamily=MONO,
            color='#999999', rotation=90)

    plt.tight_layout()
    plt.savefig('images/ch10_training.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    training_diagram()
    print('ch10 done')
