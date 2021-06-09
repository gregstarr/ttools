import numpy as np
import pandas
import os
import matplotlib.pyplot as plt

from ttools import tec as ttec, io, utils, plotting, config, satellite, compare


def run_single(date):
    # get tec data, run trough detection algo
    bg_est_shape = (1, 17, 11)
    one_h = np.timedelta64(1, 'h')
    tec_start = date - np.floor(bg_est_shape[0] / 2) * one_h
    tec_end = date + np.timedelta64(1, 'D')
    tec, times, ssmlon, n = io.get_tec_data(tec_start, tec_end)
    arb, _ = io.get_arb_data(tec_start, tec_end)
    x, times = ttec.preprocess_interval(tec, times, bg_est_shape=bg_est_shape)

    return x, arb


if __name__ == "__main__":
    start_date = np.datetime64("2014-01-01")
    end_date = np.datetime64("2020-01-01")
    time_range_days = (end_date - start_date).astype('timedelta64[D]').astype(int)
    offsets = np.random.randint(0, time_range_days, 5)
    dates = start_date + offsets.astype('timedelta64[D]')

    x = []
    arb = []
    for date in dates:
        xi, arbi = run_single(date)
        x.append(xi)
        arb.append(arbi)
    x = np.concatenate(x, axis=0)
    arb = np.concatenate(arb, axis=0)

    arb_offset = -3
    prior_weight_max = 1
    model_mlat = arb + arb_offset
    prior = abs(config.mlat_grid[None, :, :] - model_mlat[:, None, :]) ** 2
    prior -= np.min(prior, axis=1, keepdims=True)
    prior = prior_weight_max * prior / np.max(prior)

    ypad = np.pad(x, ((0, 0), (1, 0), (0, 0)), 'edge')
    gy = np.diff(ypad, axis=1)
    gy[np.isnan(gy)] = 0
    gy = np.median(utils.extract_patches(gy, (1, 3, 3)), axis=(-1, -2, -3))
    p = utils.extract_patches(100 * abs(gy), (1, 5, 1))
    ex = np.exp(p - np.nanmax(p, axis=(-1, -2, -3), keepdims=True))
    ex = ex / np.nansum(ex, axis=(-1, -2, -3), keepdims=True)
    softmax = ex[:, :, :, 0, 2, 0]
    gy = np.nansum(utils.extract_patches(gy, (1, 5, 1)), axis=(-1, -2, -3)) * softmax
    gy = np.mean(utils.extract_patches(gy, (1, 1, 3)), axis=(-1, -2, -3))

    xpad = np.pad(x, ((0, 0), (0, 0), (1, 0)), 'wrap')
    gx = np.diff(xpad, axis=2)
    gx[np.isnan(gx)] = 0
    gx = np.median(utils.extract_patches(gx, (1, 3, 3)), axis=(-1, -2, -3))
    p = utils.extract_patches(100 * abs(gx), (1, 1, 7))
    ex = np.exp(p - np.nanmax(p, axis=(-1, -2, -3), keepdims=True))
    ex = ex / np.nansum(ex, axis=(-1, -2, -3), keepdims=True)
    softmax = ex[:, :, :, 0, 0, 3]
    gx = np.nansum(utils.extract_patches(gx, (1, 1, 7)), axis=(-1, -2, -3)) * softmax
    gx = np.mean(utils.extract_patches(gx, (1, 3, 1)), axis=(-1, -2, -3))

    tv = np.cumsum(gx, axis=2) + np.cumsum(-1 * gx[:, :, ::-1], axis=2)[:, :, ::-1]
    tv += np.cumsum(gy, axis=1) + np.cumsum(-1 * gy[:, ::-1, :], axis=1)[:, ::-1, :]
    tv /= 2

    xp = x + prior + .25 * tv
    xp = np.nanmean(utils.extract_patches(xp, (1, 5, 5)), axis=(-1, -2, -3))
    xp2 = np.nanmedian(utils.extract_patches(x, (1, 3, 3)), axis=(-1, -2, -3)) + prior
    xp3 = np.nanmedian(utils.extract_patches(x, (1, 3, 3)), axis=(-1, -2, -3)) + prior + .25 * tv

    fig, ax = plt.subplots(1, 2, subplot_kw={'polar': True}, tight_layout=True)
    plotting.polar_pcolormesh(ax[0], config.mlat_grid, config.mlt_grid, tv[0], vmin=-.5, vmax=.5, cmap='coolwarm')
    plotting.polar_pcolormesh(ax[1], config.mlat_grid, config.mlt_grid, prior[0])
    plotting.format_polar_mag_ax(ax)

    fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True}, tight_layout=True)
    plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, x[0], vmin=-.5, vmax=.5, cmap='coolwarm')
    plotting.format_polar_mag_ax(ax)

    fig, ax = plt.subplots(1, 3, subplot_kw={'polar': True}, tight_layout=True)
    plotting.polar_pcolormesh(ax[0], config.mlat_grid, config.mlt_grid, xp[0], vmin=-.5, vmax=.5, cmap='coolwarm')
    ax[0].plot(np.pi * (config.mlt_vals - 6) / 12, 90 - arb[0], 'k--')
    plotting.polar_pcolormesh(ax[1], config.mlat_grid, config.mlt_grid, xp2[0], vmin=-.5, vmax=.5, cmap='coolwarm')
    ax[1].plot(np.pi * (config.mlt_vals - 6) / 12, 90 - arb[0], 'k--')
    plotting.polar_pcolormesh(ax[2], config.mlat_grid, config.mlt_grid, xp3[0], vmin=-.5, vmax=.5, cmap='coolwarm')
    ax[2].plot(np.pi * (config.mlt_vals - 6) / 12, 90 - arb[0], 'k--')
    plotting.format_polar_mag_ax(ax)

    plt.show()
