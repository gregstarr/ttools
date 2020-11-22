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


def plot_swarm_trough_detection(ax, swarm_segment):
    ax.plot(swarm_segment.data['MLat'], swarm_segment.data['smooth_dne'])
    ax.plot(swarm_segment.data['MLat'], swarm_segment.data['dne'], 'k.', ms=1)
    if swarm_segment.trough:
        ax.plot([swarm_segment.pwall_lat, swarm_segment.ewall_lat], [0, 0], 'r', linestyle='solid', marker='|', ms=10)
        ax.plot(swarm_segment.min_lat, swarm_segment.min_dne, 'rx')


def plot_swarm_trough_detections_polar(ax, swarm_segments):
    for segment in swarm_segments:
        if segment.trough:
            ax.plot((segment.data['MLT'][[segment.pwall_ind, segment.ewall_ind]] - 6) * np.pi / 12,
                    [90 - segment.pwall_lat, 90 - segment.ewall_lat],
                    'r-')


def polar_pcolormesh(ax, mlat, mlt, value, **kwargs):
    r = 90 - mlat
    t = (mlt - 6) * np.pi / 12
    return ax.pcolormesh(t - np.pi/360, r + .5, value, **kwargs)
