import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import stats
from skimage.morphology import binary_dilation, ball

from ttools import config, plotting, io, tec
import clock_angle as ca


MLT_DITHER = .025
MLT_BOUNDS = (-12, 12)
MLT_BINS = 40

KP_DITHER = .3


def plot_param_mlt(trough, param, fin, param_bins=50, param_bounds=None, name='param', lr=False, save_dir=None):
    shp = (trough.shape[0], trough.shape[2])
    x = np.broadcast_to(param[:, None], shp)

    y = np.broadcast_to(config.mlt_vals[None, :], shp)
    y = y + np.random.randn(*y.shape) * MLT_DITHER

    if param_bounds is None:
        param_bounds = np.nanquantile(param, [.005, .995])

    trough_mask = np.any((trough * fin), axis=1)
    trough_mask *= np.isfinite(x)
    obs_mask = np.any(fin, axis=1)
    obs_mask *= np.isfinite(x)

    total_counts, *_ = np.histogram2d(x[obs_mask], y[obs_mask], bins=[param_bins, MLT_BINS], range=[param_bounds, MLT_BOUNDS])

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), tight_layout=True, gridspec_kw={'hspace': .25, 'wspace': .1, 'height_ratios': [30, 1]})
    counts, xe, ye, pcm = ax[0, 0].hist2d(x[trough_mask], y[trough_mask], bins=[param_bins, MLT_BINS], range=[param_bounds, MLT_BOUNDS], cmap='jet')
    plt.colorbar(pcm, cax=ax[1, 0], orientation='horizontal')

    xv = (xe[:-1] + xe[1:]) / 2
    yv = (ye[:-1] + ye[1:]) / 2
    xg, yg = np.meshgrid(xv, yv)
    prob = counts / total_counts
    prob[total_counts < 100] = np.nan
    pcm = ax[0, 1].pcolormesh(xg, yg, prob.T, cmap='jet', vmin=0, vmax=.9)
    plt.colorbar(pcm, cax=ax[1, 1], orientation='horizontal')

    mean_result = stats.binned_statistic(x[trough_mask], y[trough_mask], 'mean', param_bins // 2, range=param_bounds)
    bin_centers = (mean_result.bin_edges[:-1] + mean_result.bin_edges[1:]) / 2
    std_result = stats.binned_statistic(x[trough_mask], y[trough_mask], 'std', param_bins // 2, range=param_bounds)
    if lr:
        lr_result = stats.linregress(x[trough_mask], y[trough_mask])
        print(lr_result)
        line = lr_result.intercept + lr_result.slope * param_bounds
        ax[0, 1].plot(param_bounds, line, 'k--', lw=2)
        ax[0, 1].errorbar(bin_centers, mean_result.statistic, yerr=std_result.statistic, fmt='ko')
    else:
        ax[0, 1].errorbar(bin_centers, mean_result.statistic, yerr=std_result.statistic)

    fig.suptitle(f"N = {counts.sum()}")
    ax[0, 0].set_xlabel(name)
    ax[0, 1].set_xlabel(name)
    ax[1, 0].set_xlabel('count')
    ax[1, 1].set_xlabel('probability')
    ax[0, 0].set_ylabel('MLT')
    if save_dir is not None:
        fn = f"{name}_mlt_dist.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_polar_probability_param(trough, param, fin, bin_edges=None, name='param', save_dir=None):
    if bin_edges is None:
        bin_edges = np.nanquantile(param, [0, .6, .8, .95, 1])

    prob = np.empty((len(bin_edges) - 1, trough.shape[1], trough.shape[2]))
    for i in range(len(bin_edges) - 1):
        mask = (param > bin_edges[i]) & (param <= bin_edges[i + 1])
        tot_obs = np.sum(fin[mask], axis=0)
        pos_obs = np.sum(trough[mask] & fin[mask], axis=0)
        prob[i] = pos_obs / tot_obs
    max_prob = np.nanmax(prob)

    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    gs = plt.GridSpec(2, 3, width_ratios=[30, 30, 1])

    for i in range(prob.shape[0]):
        ax = fig.add_subplot(gs[i // 2, i % 2], polar=True)
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, prob[i], cmap='jet', vmin=0, vmax=max_prob)
        ax.set_title(f'({bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]', loc='left')
        plotting.format_polar_mag_ax(ax, tick_color='grey')
    cb_ax = fig.add_subplot(gs[:, -1])
    plt.colorbar(pcm, cax=cb_ax)

    if save_dir is not None:
        fn = f"{name}_polar_prob.png"
        fig.savefig(os.path.join(save_dir, fn))


def get_saps(start_time, end_time, vth=350, selem=None):
    fx, fy, _ = io.get_superdarn_data(start_time, end_time)
    theta = np.pi * config.mlt_grid / 12
    fwest = -(np.cos(theta) * fx + np.sin(theta) * fy)
    arb, _ = io.get_arb_data(start_time, end_time)
    saps = fwest > vth
    saps = binary_dilation(saps, selem=selem)
    saps = tec.remove_auroral(saps, arb, -1)

    return saps, np.isfinite(fwest)


def plot_saps_trough_occurrence(trough, saps, fin, kp, low_kp=2, high_kp=4):
    ftrough = np.where(fin, trough, np.nan)
    fsaps = np.where(fin, saps, np.nan)

    hts = f'kp >= {high_kp}'
    lts = f'kp <= {low_kp}'
    data = {}

    mask = kp <= low_kp
    nst_sum = np.nansum(ftrough[mask] * (fsaps[mask] == 0), axis=0)
    nst_n = np.nansum(np.isfinite(ftrough[mask] * (fsaps[mask] == 0)), axis=0)
    nst = np.where(nst_n > 100, nst_sum / nst_n, np.nan)
    st_sum = np.nansum(ftrough[mask] * fsaps[mask], axis=0)
    st_n = np.nansum(np.isfinite(ftrough[mask] * fsaps[mask]), axis=0)
    st = np.where(st_n > 100, st_sum / st_n, np.nan)
    at_sum = np.nansum(ftrough[mask], axis=0)
    at_n = np.nansum(np.isfinite(ftrough[mask]), axis=0)
    at = np.where(at_n > 100, at_sum / at_n, np.nan)
    data[lts] = {'no_saps_trough': nst, 'saps_trough': st, 'any_trough': at}
    vm = max(np.nanmax(nst), np.nanmax(st), np.nanmax(at))

    mask = kp >= high_kp
    nst_sum = np.nansum(ftrough[mask] * (fsaps[mask] == 0), axis=0)
    nst_n = np.nansum(np.isfinite(ftrough[mask] * (fsaps[mask] == 0)), axis=0)
    nst = np.where(nst_n > 100, nst_sum / nst_n, np.nan)
    st_sum = np.nansum(ftrough[mask] * fsaps[mask], axis=0)
    st_n = np.nansum(np.isfinite(ftrough[mask] * fsaps[mask]), axis=0)
    st = np.where(st_n > 100, st_sum / st_n, np.nan)
    at_sum = np.nansum(ftrough[mask], axis=0)
    at_n = np.nansum(np.isfinite(ftrough[mask]), axis=0)
    at = np.where(at_n > 100, at_sum / at_n, np.nan)
    data[hts] = {'no_saps_trough': nst, 'saps_trough': st, 'any_trough': at}
    vm = max(vm, np.nanmax(nst), np.nanmax(st), np.nanmax(at))

    for j, c in enumerate(data[lts].keys()):
        fig = plt.figure(figsize=(8, 4), tight_layout=True)
        gs = plt.GridSpec(1, 3, width_ratios=[30, 30, 1])
        for i, kpl in enumerate(data.keys()):
            ax = fig.add_subplot(gs[i], polar=True)
            ax.set_facecolor('grey')
            pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, data[kpl][c], cmap='jet', vmin=0, vmax=vm)
            plotting.format_polar_mag_ax(ax, tick_color='grey')
            ax.set_title(f'{kpl}', loc='left')
        cb_ax = fig.add_subplot(gs[:, -1])
        plt.colorbar(pcm, cax=cb_ax)


def plot_saps_trough_portion(trough, saps, fin, kp, low_kp=2, high_kp=4):
    ftrough = np.where(fin, trough, np.nan)
    fsaps = np.where(fin, saps, np.nan)

    data = {}
    mask = kp <= low_kp
    tsum = np.nansum(ftrough[mask], axis=0)
    bsum = np.nansum(fsaps[mask] * ftrough[mask], axis=0)
    prob = bsum / tsum
    prob[tsum < 100] = np.nan
    data[f'kp <= {low_kp}'] = prob
    vm = np.nanmax(prob)

    mask = kp >= high_kp
    tsum = np.nansum(ftrough[mask], axis=0)
    bsum = np.nansum(fsaps[mask] * ftrough[mask], axis=0)
    prob = bsum / tsum
    prob[tsum < 100] = np.nan
    data[f'kp >= {high_kp}'] = prob
    vm = max(vm, np.nanmax(prob))

    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    gs = plt.GridSpec(1, 3, width_ratios=[30, 30, 1])
    for i, kpl in enumerate(data.keys()):
        ax = fig.add_subplot(gs[0, i], polar=True)
        ax.set_facecolor('grey')
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, data[kpl], cmap='jet', vmin=0, vmax=vm)
        plotting.format_polar_mag_ax(ax, tick_color='grey')
        ax.set_title(kpl, loc='left')
    cb_ax = fig.add_subplot(gs[0, 2])
    plt.colorbar(pcm, cax=cb_ax)


if __name__ == "__main__":
    trough_data = np.load("E:\\dataset.npz")
    trough = trough_data['trough']
    x = trough_data['x']
    kp = io.get_kp(trough_data['time'])
    # kp += np.random.randn(*kp.shape) * KP_DITHER

    RECALCULATE = True
    if RECALCULATE:
        saps = np.zeros_like(trough)
        fins = np.zeros_like(trough)
        batch_size = 10000
        for batch in range(int(np.ceil(trough.shape[0] / batch_size))):
            print(batch, int(np.ceil(trough.shape[0] / batch_size)))
            start_time = trough_data['time'][batch * batch_size]
            end_time = min(start_time + (batch_size - 1) * np.timedelta64(1, 'h'), trough_data['time'][-1]) + np.timedelta64(1, 'h')
            pre = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=bool)
            selem = np.stack((pre, np.ones((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool)), axis=0)
            sap, fin = get_saps(start_time, end_time, 400, selem=selem)
            saps[batch * batch_size:batch * batch_size + sap.shape[0]] = sap
            fins[batch * batch_size:batch * batch_size + fin.shape[0]] = fin
        np.save("E:\\saps.npy", saps)
        np.save("E:\\fins.npy", fins)
    else:
        saps = np.load("E:\\saps.npy")
        fins = np.load("E:\\fins.npy")

    fins &= np.isfinite(x)
    plot_saps_trough_occurrence(trough, saps, fins, kp, 2, 4)
    plot_saps_trough_portion(trough, saps, fins, kp, 2, 4)
    plt.show()
