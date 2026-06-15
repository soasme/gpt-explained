import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; MONO='monospace'

def softmax_plot():
    z = np.array([2.0, 1.0, 0.1])
    e = np.exp(z - z.max())
    probs = e / e.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
    labels = ['z=2.0', 'z=1.0', 'z=0.1']
    x = np.arange(3)

    ax1.set_facecolor('white')
    ax1.bar(x, z, color=[BLUE, GREEN, '#AAAAAA'], width=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontfamily=MONO, fontsize=11)
    ax1.set_title('Raw logits', fontfamily=MONO, fontsize=12, color=DARK)
    ax1.set_ylabel('value', fontfamily=MONO, fontsize=10)
    for sp in ['top','right']: ax1.spines[sp].set_visible(False)

    ax2.set_facecolor('white')
    bars = ax2.bar(x, probs, color=[BLUE, GREEN, '#AAAAAA'], width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontfamily=MONO, fontsize=11)
    ax2.set_title('After softmax (probabilities)', fontfamily=MONO, fontsize=12, color=DARK)
    ax2.set_ylabel('probability', fontfamily=MONO, fontsize=10)
    ax2.set_ylim(0, 1.0)
    for sp in ['top','right']: ax2.spines[sp].set_visible(False)
    for bar, p in zip(bars, probs):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{p:.3f}', ha='center', fontfamily=MONO, fontsize=10, color=DARK)

    plt.suptitle('Softmax: logits to probabilities', fontfamily=MONO, fontsize=12, color=DARK)
    plt.tight_layout()
    plt.savefig('images/ch02_softmax.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    softmax_plot()
    print('ch02_softmax done')
