import matplotlib.pyplot as plt
import numpy as np

DARK='#222222'; BLUE='#1177BB'; GREEN='#228811'; RED='#CC2222'; MONO='monospace'

def dot_product():
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.axhline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.axvline(0, color=DARK, lw=0.5, alpha=0.3)
    ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
    ax.tick_params(labelsize=9)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

    u = np.array([3.0, 2.0])
    v = np.array([4.0, 1.0])
    v_unit = v / np.linalg.norm(v)
    proj_len = np.dot(u, v_unit)
    proj = proj_len * v_unit

    ax.annotate('', xy=u, xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
    ax.text(u[0]+0.1, u[1]+0.1, 'u = [3, 2]', fontfamily=MONO, fontsize=11, color=BLUE)

    ax.annotate('', xy=v, xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
    ax.text(v[0]+0.1, v[1]+0.1, 'v = [4, 1]', fontfamily=MONO, fontsize=11, color=GREEN)

    ax.plot([proj[0], u[0]], [proj[1], u[1]], color=DARK, lw=1.2, linestyle='--', alpha=0.5)
    ax.annotate('', xy=proj, xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color=RED, lw=2.0))
    ax.text(proj[0]+0.1, proj[1]-0.25,
            f'u dot v = {np.dot(u,v):.0f}', fontfamily=MONO, fontsize=11, color=RED)

    sq = 0.2
    perp = np.array([-v_unit[1], v_unit[0]]) * sq
    corner = proj + perp
    ax.plot([proj[0], corner[0], corner[0]+v_unit[0]*sq],
            [proj[1], corner[1], corner[1]+v_unit[1]*sq],
            color=DARK, lw=1.0, alpha=0.5)

    ax.set_title('Dot product as scalar projection', fontfamily=MONO, fontsize=12, color=DARK)
    plt.tight_layout()
    plt.savefig('images/ch02_dot_product.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    dot_product()
    print('ch02_dot_product done')
