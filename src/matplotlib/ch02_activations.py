import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; MONO='monospace'

def activations_plot():
    x = np.linspace(-3, 3, 300)
    relu = np.maximum(0, x)
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.set_facecolor('white')
    ax.axhline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.axvline(0, color=DARK, lw=0.5, alpha=0.3)

    ax.plot(x, relu, color=BLUE, lw=2.5, label='ReLU')
    ax.plot(x, gelu, color=GREEN, lw=2.5, label='GELU', linestyle='--')

    ax.legend(fontsize=12, prop={'family': MONO})
    ax.set_xlabel('x', fontfamily=MONO, fontsize=12)
    ax.set_ylabel('activation(x)', fontfamily=MONO, fontsize=12)
    ax.set_title('Activation functions: ReLU vs GELU', fontfamily=MONO, fontsize=12, color=DARK)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig('images/ch02_activations.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    activations_plot()
    print('ch02_activations done')
