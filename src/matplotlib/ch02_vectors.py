import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; RED='#CC2222'; MONO='monospace'

def vectors():
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor('white')
    ax.axhline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.axvline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.set_xlim(-1, 6); ax.set_ylim(-3, 6)
    ax.set_xticks(range(-1, 7)); ax.set_yticks(range(-3, 7))
    ax.tick_params(labelsize=9)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

    ax.annotate('', xy=(3,4), xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
    ax.text(3.15, 4.1, 'u = [3, 4]', fontsize=12, fontfamily=MONO, color=BLUE)

    ax.annotate('', xy=(1,-2), xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
    ax.text(1.15, -2.2, 'v = [1, -2]', fontsize=12, fontfamily=MONO, color=GREEN)

    ax.annotate('', xy=(4,2), xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=RED, lw=2.5))
    ax.text(4.15, 2.1, 'u+v = [4, 2]', fontsize=12, fontfamily=MONO, color=RED)

    ax.annotate('', xy=(0,0), xytext=(3,4),
        arrowprops=dict(arrowstyle='<->', color=BLUE, lw=1.5, linestyle='dashed'))
    ax.text(-0.9, 2.2, '||u|| = 5', fontsize=11, fontfamily=MONO, color=BLUE)

    ax.set_title('Vector addition and norm', fontsize=12, fontfamily=MONO, color=DARK)
    plt.tight_layout()
    plt.savefig('images/ch02_vectors.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    vectors()
    print('ch02_vectors done')
