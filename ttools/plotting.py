import numpy as np
import matplotlib.pyplot as plt


def format_polar_mag_ax(ax):
    if isinstance(ax, np.ndarray):
        for a in ax.flatten():
            format_polar_mag_ax(a)
    else:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.set_ylim(0, 60)
        ax.set_xticks(np.arange(8) * np.pi/4)
        ax.set_xticklabels((np.arange(8) * 3 + 6) % 24)
        ax.set_yticks([10, 20, 30, 40, 50])
        ax.set_yticklabels([80, 70, 60, 50, 40])
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=True, labelleft=True, width=0, length=0)
        ax.set_rlabel_position(80)
        ax.grid(True)


def plot_swarm_troughs_line(ax, mlat, dne, smooth_dne, trough):
    ax.plot(mlat, smooth_dne)
    ax.plot(mlat, dne, 'k.', ms=1)
    if trough['trough']:
        ax.plot(trough[['e1_mlat', 'e2_mlat']], [0, 0], 'r', linestyle='solid', marker='|', ms=10)
        ax.plot(trough['min_mlat'], trough['min_dne'], 'rx')


def plot_swarm_troughs_polar(ax, troughs):
    for i, trough in troughs.iterrows():
        ax.plot((trough[['seg_e1_mlt', 'seg_e2_mlt']] - 6) * np.pi / 12,
                90 - trough[['seg_e1_mlat', 'seg_e2_mlat']],
                'k--')
        if trough['trough']:
            ax.plot((trough[['e1_mlt', 'e2_mlt']] - 6) * np.pi / 12, 90 - trough[['e1_mlat', 'e2_mlat']], 'r-')


def polar_pcolormesh(ax, mlat, mlt, value, **kwargs):
    r = 90 - mlat
    t = (mlt - 6) * np.pi / 12
    return ax.pcolormesh(t - np.pi/360, r + .5, value, shading='auto', **kwargs)
