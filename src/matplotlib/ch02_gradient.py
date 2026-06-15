import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; RED='#CC2222'; MONO='monospace'

def gradient_plot():
    theta = np.linspace(-1, 7, 300)
    L = (theta - 3) ** 2

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.set_facecolor('white')
    ax.plot(theta, L, color=BLUE, lw=2.5, label='L = (theta - 3)^2')

    t = 0.0
    eta = 0.1
    steps = []
    for _ in range(12):
        steps.append((t, (t-3)**2))
        grad = 2 * (t - 3)
        t = t - eta * grad

    xs, ys = zip(*steps)
    ax.plot(xs, ys, 'o-', color=RED, markersize=7, lw=1.5, label='gradient descent steps')
    ax.annotate('start', xy=(xs[0], ys[0]), xytext=(xs[0]-0.5, ys[0]+1),
                fontfamily=MONO, fontsize=10, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
    ax.annotate('minimum', xy=(3, 0), xytext=(3.5, 1.5),
                fontfamily=MONO, fontsize=10, color=DARK,
                arrowprops=dict(arrowstyle='->', color=DARK, lw=1.2))

    ax.set_xlabel('theta', fontfamily=MONO, fontsize=12)
    ax.set_ylabel('L(theta)', fontfamily=MONO, fontsize=12)
    ax.set_title('Gradient descent converges to the minimum', fontfamily=MONO, fontsize=12, color=DARK)
    ax.legend(fontsize=10, prop={'family': MONO})
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig('images/ch02_gradient.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    gradient_plot()
    print('ch02_gradient done')
