import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas

from ttools import config, io, plotting

MLT_DITHER = .025
MLT_BOUNDS = (-12, 12)
MLT_BINS = 40

KP_DITHER = 0.3


if __name__ == "__main__":
    trough_data = np.load("E:\\dataset.npz")
    trough = trough_data['trough']
    x = trough_data['x']
    xi = x.copy()
    xi[~trough] = np.inf
    min_mlat = config.mlat_vals[np.argmin(xi, axis=1)]
    depth = np.min(xi, axis=1)
    min_mlat[~np.isfinite(depth)] = np.nan
    depth[~np.isfinite(depth)] = np.nan
    width = np.sum(trough, axis=1).astype(float)
    width[width == 0] = np.nan
    kp = io.get_kp(trough_data['time'])
    kp += np.random.randn(*kp.shape) * KP_DITHER

    tfht = trough.reshape((-1, 24, 60, 180)).sum(axis=1) >= 4
    fig, ax = plt.subplots(1, 4, subplot_kw={'polar': True})
    for i in range(4):
        plotting.polar_pcolormesh(ax[i], config.mlat_grid, config.mlt_grid, tfht[tfht[:, 10 + i * 12, 90]].mean(axis=0), cmap='jet', vmin=0, vmax=1)
    plotting.format_polar_mag_ax(ax)
    plt.show()
