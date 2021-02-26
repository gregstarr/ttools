"""
For large portion of dataset:
    - average dlogtec in mlat mlon bins
    - try different bg_est_shapes: 3, 5, 7
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

from ttools import io, rbf_inversion, plotting
from ttools.config import mlat_grid, mlt_grid


def run_interval(start_date, end_date, bg_size, bins):
    tec, times, ssmlon, n_samples = io.get_tec_data(start_date, end_date)
    z, t = rbf_inversion.preprocess_interval(tec, times, bg_est_shape=(1, bg_size, bg_size))
    mlon_grid = (15 * mlt_grid[None, :, :] - 180 + ssmlon[:, None, None]) % 360
    mlon_grid[mlon_grid > 359.5] -= 360
    mlat_grid_ext = mlat_grid[None, :, :] * np.ones_like(t, dtype=float)[:, None, None]
    fin_mask = np.isfinite(z)
    sum_r = binned_statistic_2d(mlat_grid_ext[fin_mask], mlon_grid[fin_mask], z[fin_mask], statistic='sum', bins=bins)
    cnt_r = binned_statistic_2d(mlat_grid_ext[fin_mask], mlon_grid[fin_mask], z[fin_mask], statistic='count', bins=bins)
    return sum_r.statistic, cnt_r.statistic


dt = np.timedelta64(6, 'M').astype("timedelta64[s]")
bins = [np.arange(29.5, 90), np.arange(-.5, 360)]
x_vals = np.mean(np.column_stack((bins[0][:-1], bins[0][1:])), axis=1)
y_vals = np.mean(np.column_stack((bins[1][:-1], bins[1][1:])), axis=1)
x_grid, y_grid = np.meshgrid(y_vals, x_vals)
output_fn = "E:\\tec_data\\tec_artifact.npz"

bgs = [3, 5, 7, 9]
n = 14
artifacts = {}
fig, axs = plt.subplots(1, len(bgs), subplot_kw={'projection': 'polar'}, figsize=(16, 12))
for bg, ax in zip(bgs, axs):
    print(bg)
    start_date = np.datetime64("2013-01-01T00:00:00")
    sum_acc = np.zeros_like(x_grid)
    cnt_acc = np.zeros_like(x_grid)
    for i in range(n):
        print(i)
        s, c = run_interval(start_date, start_date + dt, bg, bins)
        start_date += dt
        sum_acc += s
        cnt_acc += c

    mean = sum_acc / cnt_acc
    artifacts[str(bg)] = mean
    pcm = plotting.polar_pcolormesh(ax, y_grid, x_grid, mean, cs='mlon', cmap='coolwarm', vmin=-.1, vmax=.1)
np.savez(output_fn, **artifacts, mlat_vals=x_vals, mlon_vals=y_vals)
plt.show()
