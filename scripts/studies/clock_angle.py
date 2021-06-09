import numpy as np
import bottleneck as bn
import pandas
import os
import matplotlib.pyplot as plt

from ttools import config, io, plotting


def plot_param_hist(param, bins=40, bounds=None, name='param', save_dir=None, time_mask=None):
    if time_mask is None:
        time_mask = np.ones(param.shape[0], dtype=bool)

    if bounds is None:
        bounds = np.nanquantile(param, [.01, .99])

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    counts, *_ = ax.hist(param[time_mask], bins=bins, range=bounds)
    ax.grid()
    ax.set_title(f"N = {counts.sum()}")
    ax.set_xlabel(name)
    ax.set_ylabel('count')
    if save_dir is not None:
        fn = f"{name}_dist.png"
        fig.savefig(os.path.join(save_dir, fn))


if __name__ == "__main__":
    # Load trough dataset
    trough_data = np.load("E:\\dataset.npz")
    trough = trough_data['trough']

    x = trough_data['x']
    # Load Omni
    omni = io.get_omni_data()
    bmag = omni['b_mag'][trough_data['time']].values
    e_field = omni['e_field'][trough_data['time']].values
    by = omni['by_gsm'][trough_data['time']].values
    bz = omni['bz_gsm'][trough_data['time']].values
    # bmag += np.random.randn(*bmag.shape) * .025
    bmag_mask = (bmag >= 4) & (bmag <= 6)
    clock_angle = np.arctan2(by, bz)
    clock_angle[clock_angle < 0] += 2 * np.pi
    clock_angle[clock_angle > np.pi * 31 / 16] -= 2 * np.pi
    ca_bins = np.arange(9) * np.pi / 4 - np.pi / 8
    for i in range(8):
        fig, ax = plt.subplots(subplot_kw={'polar': True})
        mask = (clock_angle >= ca_bins[i]) & (clock_angle < ca_bins[i + 1]) & bmag_mask
        fig.suptitle(f"N = {mask.sum()}")
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, trough[mask].sum(axis=0), cmap='jet')
        plt.colorbar(pcm)
        plotting.format_polar_mag_ax(ax)
    plt.show()
