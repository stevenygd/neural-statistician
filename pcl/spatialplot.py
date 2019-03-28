import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def visualize_point_clouds(pts, gtr, idx, pert_order=[0, 1, 2]):
    pts = pts.cpu().detach().numpy()[:, pert_order]
    gtr = gtr.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Sample:%s" % (idx))
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Ground Truth:%s" % (idx))
    ax2.scatter(gtr[:, 0], gtr[:, 1], gtr[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    ret = np.array(fig.canvas.renderer._renderer)
    ret = np.transpose(ret, (2, 0, 1))

    plt.close()
    return ret


def grid(inputs, samples, summaries=None, save_path=None, ncols=10):

    inputs = inputs.data.cpu().numpy()
    samples = samples.data.cpu().numpy()
    if summaries is not None:
        summaries = summaries.data.cpu().numpy()
    fig, axs = plt.subplots(nrows=2, ncols=ncols, figsize=(ncols, 2))

    def plot_single(ax, points, s, color):
        ax.scatter(points[:, 0], points[:, 1], s=s,  color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, 27])
        ax.set_ylim([0, 27])
        ax.set_aspect('equal', adjustable='box')

    for i in range(ncols):
        # fill one column of subplots per loop iteration
        plot_single(axs[0, i], inputs[i], s=5, color='C0')
        plot_single(axs[1, i], samples[i], s=5, color='C1')
        if summaries is not None:
            plot_single(axs[1, i], summaries[i], s=10, color='C2')

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()

    fig.canvas.draw()
    ret = np.array(fig.canvas.renderer._renderer)
    ret = np.transpose(ret, (2, 0, 1))

    if save_path is not None:
        fig.savefig(save_path)

    return ret

