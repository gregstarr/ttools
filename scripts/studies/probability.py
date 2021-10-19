import numpy as np
import bottleneck as bn
import pandas
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats

from ttools import config, io, plotting


MONTH_INDEX = {
    'winter': [0, 1, 10, 11],
    'equinox': [2, 3, 8, 9],
    'summer': [4, 5, 6, 7],
}

DITHER = {
    'mlt': .02,
    'mlat': .01,
    'width': .5,
    'kp': .1,
    'log_kp': .1,
    'by': .05,
    'bz': .05,
    'dst': 0.5,
    'density': .15,
    'log_density': .03,
    'f107': 3,
    'log_f107': .03,
    'pressure': .15,
    'log_pressure': .02,
    'log_speed': .02,
    'log_newell': .1,
    'log_temp': .01,
    'speed': 1,
    'temp': 10,
    'e_field': .01,
}

BINS = {
    'width': 40,
    'mlt': 40,
    'depth': 50,
    'mlat': 40,
}

LOG = ['kp', 'newell', 'density', 'f107', 'pressure', 'speed', 'temp']

WIDTH_BOUNDS = [1, 12]
DEPTH_BOUNDS = [-1, 0]
MLT_BOUNDS = [-12, 12]
MLAT_BOUNDS = [40, 80]


def plot_polar_probability_season(time, trough, fin, save_dir=None):
    months = (time.astype('datetime64[M]') - time.astype('datetime64[Y]')).astype(int)

    prob = {}
    for season, mo in MONTH_INDEX.items():
        mask = np.zeros_like(months, dtype=bool)
        for m in mo:
            mask |= months == m
        season_trough = trough[mask].astype(float)
        season_trough[~fin[mask]] = np.nan
        prob[season] = bn.nanmean(trough[mask], axis=0)
        prob[season][np.sum(fin[mask], axis=0) < 100] = np.nan
    max_prob = max([np.nanmax(p) for p in prob.values()])

    fig = plt.figure(figsize=(16, 6), tight_layout=True)
    gs = plt.GridSpec(1, 4, width_ratios=[20, 20, 20, 1])

    for i, (season, p) in enumerate(prob.items()):
        ax = fig.add_subplot(gs[i], polar=True)
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, p, cmap='jet', vmin=0, vmax=max_prob)
        ax.set_title(season, loc='left')
        plotting.format_polar_mag_ax(ax, tick_color='grey')

    cb_ax = fig.add_subplot(gs[3])
    plt.colorbar(pcm, cax=cb_ax)

    if save_dir is not None:
        fn = f"polar_prob.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_average_param_season(time, x_param, y_param, bins=50, bounds=None, x_name='xparam', y_name='yparam', save_dir=None):
    months = (time.astype('datetime64[M]') - time.astype('datetime64[Y]')).astype(int)
    if bounds is None:
        bounds = np.nanquantile(x_param, [.005, .995])

    prob = {}
    for season, mo in MONTH_INDEX.items():
        mask = np.zeros_like(months, dtype=bool)
        for m in mo:
            mask |= months == m
        mask = mask[:, None] & np.isfinite(x_param) & np.isfinite(y_param)
        mean_result = stats.binned_statistic(x_param[mask], y_param[mask], 'mean', bins, bounds)
        std_result = stats.binned_statistic(x_param[mask], y_param[mask], 'std', bins, bounds)
        prob[season] = {'mean': mean_result.statistic, 'std': std_result.statistic}
    x = (mean_result.bin_edges[:-1] + mean_result.bin_edges[1:]) / 2
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))

    for i, (season, s) in enumerate(prob.items()):
        ax[i].errorbar(x[::2], s['mean'][::2], s['std'][::2], fmt='k.')
        ax[i].plot(x, s['mean'], 'k-')
        ax[i].set_title(season, loc='left')
        ax[i].grid()
    ax[0].set_ylabel(y_name)
    ax[1].set_xlabel(x_name)

    if save_dir is not None:
        fn = f"season_{x_name}_{y_name}.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_polar_probability_param(trough, param, bins, rows=2):
    if isinstance(bins, list) or isinstance(bins, np.ndarray):
        bin_edges = bins
    else:
        bin_edges = np.linspace(np.nanmin(param), np.nanmax(param), bins + 1)

    prob = {}
    for i in range(len(bin_edges) - 1):
        bin = (bin_edges[i], bin_edges[i + 1])
        mask = (param > bin[0]) & (param <= bin[1])
        prob[bin] = bn.nanmean(trough[mask], 0)
    max_prob = np.nanmax([np.nanmax(p) for p in prob.values()])

    row_size = int(np.ceil((len(bin_edges) - 1) / rows))

    fig = plt.figure(figsize=(16, 6), tight_layout=True)
    gs = plt.GridSpec(rows, row_size + 1, width_ratios=[20] * row_size + [1])

    for i, (bin, p) in enumerate(prob.items()):
        ax = fig.add_subplot(gs[i // row_size, i % row_size], polar=True)
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, p, cmap='jet', vmin=0, vmax=max_prob)
        ax.set_title(f'({bin[0]:.2f}, {bin[1]:.2f}]', loc='left')
        plotting.format_polar_mag_ax(ax, tick_color='grey')

    cb_ax = fig.add_subplot(gs[:, -1])
    plt.colorbar(pcm, cax=cb_ax)


def plot_season_mlt_probability(time, trough, fin, n_season_bins=40):
    x = (time.astype('datetime64[s]') - time.astype('datetime64[Y]')).astype('timedelta64[s]').astype(float) / (60 * 60 * 24)
    x = np.broadcast_to(x[:, None], (trough.shape[0], trough.shape[2]))
    x = x + np.random.randn(*x.shape) * .25
    y = np.broadcast_to(config.mlt_vals[None, :], (trough.shape[0], trough.shape[2]))
    y = y + np.random.randn(*y.shape) * DITHER['mlt']

    trough_mask = np.any((trough * fin), axis=1)
    obs_mask = np.any(fin, axis=1)

    total_counts, *_ = np.histogram2d(x[obs_mask], y[obs_mask], bins=[n_season_bins, BINS['mlt']], range=[(0, 365), MLT_BOUNDS])

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), tight_layout=True, gridspec_kw={'hspace': .25, 'wspace': .1, 'height_ratios': [30, 1]})
    counts, xe, ye, pcm = ax[0, 0].hist2d(x[trough_mask], y[trough_mask], bins=[n_season_bins, BINS['mlt']], range=[(0, 365), MLT_BOUNDS], cmap='jet')
    plt.colorbar(pcm, cax=ax[1, 0], orientation='horizontal')

    xv = (xe[:-1] + xe[1:]) / 2
    yv = (ye[:-1] + ye[1:]) / 2
    xg, yg = np.meshgrid(xv, yv)
    prob = counts / total_counts
    prob[total_counts < 100] = np.nan
    pcm = ax[0, 1].pcolormesh(xg, yg, prob.T, cmap='jet')
    plt.colorbar(pcm, cax=ax[1, 1], orientation='horizontal')

    fig.suptitle(f"N = {counts.sum()}")
    ax[0, 0].set_xlabel('day')
    ax[0, 1].set_xlabel('day')
    ax[1, 0].set_xlabel('count')
    ax[1, 1].set_xlabel('probability')
    ax[0, 0].set_ylabel('MLT')


def plot_param_mlt(trough, param, fin, param_bins=50, param_bounds=None, name='param', save_dir=None):
    shp = (trough.shape[0], trough.shape[2])
    x = np.broadcast_to(param[:, None], shp)

    y = np.broadcast_to(config.mlt_vals[None, :], shp)
    y = y + np.random.randn(*y.shape) * DITHER['mlt']

    if param_bounds is None:
        param_bounds = np.nanquantile(param, [.005, .995])

    trough_mask = np.any((trough * fin), axis=1)
    trough_mask *= np.isfinite(x)
    obs_mask = np.any(fin, axis=1)
    obs_mask *= np.isfinite(x)

    total_counts, *_ = np.histogram2d(x[obs_mask], y[obs_mask], bins=[param_bins, BINS['mlt']], range=[param_bounds, MLT_BOUNDS])

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), tight_layout=True, gridspec_kw={'hspace': .25, 'wspace': .1, 'height_ratios': [30, 1]})
    counts, xe, ye, pcm = ax[0, 0].hist2d(x[trough_mask], y[trough_mask], bins=[param_bins, BINS['mlt']], range=[param_bounds, MLT_BOUNDS], cmap='jet')
    plt.colorbar(pcm, cax=ax[1, 0], orientation='horizontal')

    xv = (xe[:-1] + xe[1:]) / 2
    yv = (ye[:-1] + ye[1:]) / 2
    xg, yg = np.meshgrid(xv, yv)
    prob = counts / total_counts
    prob[total_counts < 100] = np.nan
    pcm = ax[0, 1].pcolormesh(xg, yg, prob.T, cmap='jet', vmin=0, vmax=.9)
    plt.colorbar(pcm, cax=ax[1, 1], orientation='horizontal')

    fig.suptitle(f"N = {counts.sum()}")
    ax[0, 0].set_xlabel(name)
    ax[0, 1].set_xlabel(name)
    ax[1, 0].set_xlabel('count')
    ax[1, 1].set_xlabel('probability')
    ax[0, 0].set_ylabel('MLT')
    if save_dir is not None:
        fn = f"{name}_mlt_dist.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_param_mlat(trough, param, fin, param_bins=50, param_bounds=None, name='param', norm=None, mlt_center=0, mlt_width=1.5, save_dir=None):
    shp = (trough.shape[0], trough.shape[1])
    x = np.broadcast_to(param[:, None], shp)
    y = np.broadcast_to(config.mlat_vals[None, :], shp)
    y = y + np.random.randn(*y.shape) * DITHER['mlat']

    if param_bounds is None:
        param_bounds = np.nanquantile(param, [.005, .995])

    mlt_mask = abs(config.mlt_vals - mlt_center) <= mlt_width
    trough_mask = np.any((trough * fin)[:, :, mlt_mask], axis=-1)
    trough_mask *= np.isfinite(x)

    obs_mask = np.any(fin[:, :, mlt_mask], axis=-1)
    obs_mask *= np.isfinite(x)

    total_counts, *_ = np.histogram2d(x[obs_mask], y[obs_mask], bins=[param_bins, BINS['mlat']], range=[param_bounds, MLAT_BOUNDS])

    fig, ax = plt.subplots(2, 2, figsize=(8, 4), tight_layout=True, gridspec_kw={'hspace': .25, 'wspace': .1, 'height_ratios': [30, 1]})
    counts, xe, ye, pcm = ax[0, 0].hist2d(x[trough_mask], y[trough_mask], bins=[param_bins, BINS['mlat']], range=[param_bounds, MLAT_BOUNDS], cmap='jet', norm=norm)
    plt.colorbar(pcm, cax=ax[1, 0], orientation='horizontal')

    xv = (xe[:-1] + xe[1:]) / 2
    yv = (ye[:-1] + ye[1:]) / 2
    xg, yg = np.meshgrid(xv, yv)
    prob = counts / total_counts
    prob[total_counts < 100] = np.nan
    pcm = ax[0, 1].pcolormesh(xg, yg, prob.T, cmap='jet')
    plt.colorbar(pcm, cax=ax[1, 1], orientation='horizontal')

    fig.suptitle(f"N = {counts.sum()} || MLT = {mlt_center}")
    ax[0, 0].set_xlabel(name)
    ax[0, 1].set_xlabel(name)
    ax[1, 0].set_xlabel('count')
    ax[1, 1].set_xlabel('probability')
    ax[0, 0].set_ylabel('MLAT')
    if save_dir is not None:
        fn = f"{name}_mlat_dist{mlt_center % 24:d}{'_norm' if norm is not None else ''}.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_lparam_tparam(l_param, t_param, lparam_bins=50, tparam_bins=50, tname='tparam', lname='lparam', mlt_center=0, mlt_width=1.5, save_dir=None):
    mlt_mask = abs(config.mlt_vals - mlt_center) <= mlt_width
    x = np.broadcast_to(t_param[:, None], l_param.shape)

    tparam_bounds = np.nanquantile(t_param, [.01, .99])
    lparam_bounds = np.nanquantile(l_param, [.01, .99])

    obs_mask = np.isfinite(t_param)[:, None] & np.isfinite(l_param) & mlt_mask[None, :]

    fig, ax = plt.subplots(2, 2, figsize=(12, 6), tight_layout=True, gridspec_kw={'hspace': .25, 'wspace': .1, 'height_ratios': [30, 1]})
    counts, xe, ye, pcm = ax[0, 0].hist2d(x[obs_mask], l_param[obs_mask], bins=[tparam_bins, lparam_bins], range=[tparam_bounds, lparam_bounds], cmap='jet')
    plt.colorbar(pcm, cax=ax[1, 0], orientation='horizontal')

    total_counts = counts.sum(axis=1, keepdims=True)
    xv = (xe[:-1] + xe[1:]) / 2
    yv = (ye[:-1] + ye[1:]) / 2
    xg, yg = np.meshgrid(xv, yv)
    prob = counts / total_counts
    prob[counts < 100] = np.nan
    pcm = ax[0, 1].pcolormesh(xg, yg, prob.T, cmap='jet')
    plt.colorbar(pcm, cax=ax[1, 1], orientation='horizontal')

    fig.suptitle(f"N = {counts.sum()} || MLT = {mlt_center}")
    ax[0, 0].set_xlabel(tname)
    ax[0, 1].set_xlabel(tname)
    ax[1, 0].set_xlabel('count')
    ax[1, 1].set_xlabel('probability density')
    ax[0, 0].set_ylabel(lname)
    if save_dir is not None:
        fn = f"{tname}_{lname}_dist.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_width_depth(width, depth, time_mask, mlt_mask, file_extra=None, title_extra=None, save_dir=None, norm=None):
    x = width[time_mask][:, mlt_mask]
    y = depth[time_mask][:, mlt_mask]

    mask = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    (counts, *_, pcm) = ax.hist2d(x[mask], y[mask], bins=[BINS['width'], BINS['depth']], range=[WIDTH_BOUNDS, DEPTH_BOUNDS], cmap='jet', norm=norm)
    plt.colorbar(pcm)
    title = f"N = {counts.sum()}{' ' + title_extra if title_extra is not None else ''}"
    ax.set_title(title)
    ax.set_xlabel('width')
    ax.set_ylabel('depth')
    if save_dir is not None:
        fn = f"width_depth_dist{'_' + file_extra if file_extra is not None else ''}{'_norm' if norm is not None else ''}.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_param_hist(param, bins=50, bounds=None, name='param', save_dir=None, time_mask=None):
    if time_mask is None:
        time_mask = np.ones(param.shape[0], dtype=bool)

    if bounds is None:
        bounds = np.quantile(param[np.isfinite(param)], [.01, .99])

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    counts, *_ = ax.hist(param[time_mask], bins=bins, range=bounds)
    ax.grid()
    ax.set_title(f"N = {counts.sum()}")
    ax.set_xlabel(name)
    ax.set_ylabel('count')
    if save_dir is not None:
        fn = f"{name}_dist.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_param_mlt_set(param, set_param, save_dir, name='param', set_name='set_param', bins=50, param_bounds=None, quantiles=(0, .2, .4, .6, .8, 1)):
    edges = np.quantile(set_param[np.isfinite(set_param)], quantiles)
    bounds = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    for i, bound in enumerate(bounds):
        time_mask = (set_param >= bound[0]) & (set_param <= bound[1])
        title_extra = f"|| {set_name} = ({bound[0]:.2f}, {bound[1]:.2f})"
        plot_param_mlt(trough, param, bins, param_bounds, name, time_mask=time_mask, title_extra=title_extra,
                       file_extra=f"{set_name}_{i}", save_dir=save_dir)


def plot_width_depth_set(width, depth, set_param, set_name='set_param', quantiles=(0, .33, .66, 1)):
    edges = np.quantile(set_param[np.isfinite(set_param)], quantiles)
    bounds = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    mltb = [(-6, -3), (-3, 0), (0, 3), (3, 6)]
    for i, bound in enumerate(bounds):
        time_mask = (set_param >= bound[0]) & (set_param <= bound[1])
        for j, mlt_bounds in enumerate(mltb):
            mlt_mask = (config.mlt_vals >= mlt_bounds[0]) & (config.mlt_vals <= mlt_bounds[1])
            title_extra = f"|| {set_name} = ({bound[0]:.2f}, {bound[1]:.2f}) || MLT = ({mlt_bounds[0]:.2f}, {mlt_bounds[1]:.2f})"
            file_extra = f"{set_name}_{j}_{i}"
            plot_width_depth(width, depth, time_mask, mlt_mask, file_extra, title_extra, "C:\\Users\\Greg\\Desktop\\study plots\\width_depth")


def plot_param_min_mlat(param, mlat_min, center, width, bounds=None, nbins=20, name='param', lr=True, save_path=None):
    if bounds is None:
        bounds = np.quantile(param[np.isfinite(param)], [.005, .995])
    m = np.nanmean(mlat_min[:, abs(config.mlt_vals - center) < width], axis=1)
    fin = np.isfinite(param) & np.isfinite(m)
    result = stats.binned_statistic(param[fin], m[fin], statistic='mean', bins=nbins, range=bounds)
    std = stats.binned_statistic(param[fin], m[fin], statistic='std', bins=nbins, range=bounds)
    x = (result.bin_edges[:-1] + result.bin_edges[1:]) / 2
    y = result.statistic

    fig, ax = plt.subplots()

    if lr:
        ax.errorbar(x, y, yerr=std.statistic, fmt='o')
        lr_result = stats.linregress(param[fin], m[fin])
        print(lr_result)
        bounds_lr = lr_result.intercept + lr_result.slope * bounds
        ax.plot(bounds, bounds_lr, 'k--', lw=3)
    else:
        ax.errorbar(x, y, yerr=std.statistic)

    ax.set_title(f"MLT in [{center - width}, {center + width}]")
    ax.set_xlabel(name)
    ax.set_ylabel('MLAT of Trough Minimum')
    ax.grid()
    if save_path is not None:
        fig.savefig(save_path)


if __name__ == "__main__":
    ####################################################################################################################
    # PREPARE DATA #####################################################################################################
    ####################################################################################################################
    # Load trough dataset
    trough_data = np.load("E:\\dataset.npz")
    trough = trough_data['trough']
    x = trough_data['x']
    xi = x.copy()
    xi[~trough] = np.inf
    # Calculate trough depth / width
    min_mlat = config.mlat_vals[np.argmin(xi, axis=1)]
    depth = np.min(xi, axis=1)
    min_mlat[~np.isfinite(depth)] = np.nan
    depth[~np.isfinite(depth)] = np.nan
    width = np.sum(trough, axis=1).astype(float)
    width[width == 0] = np.nan
    # Load Omni
    omni = io.get_omni_data()
    # Assemble
    params = {
        'width': width,
        'depth': depth,
        'kp': io.get_kp(trough_data['time']),
        'newell': pandas.read_hdf("E:\\newell.h5").values,
        'bz': omni['bz_gsm'][trough_data['time']].values,
        'by': omni['by_gsm'][trough_data['time']].values,
        'bmag': omni['b_mag'][trough_data['time']].values,
        'speed': omni['plasma_speed'][trough_data['time']].values,
        'pressure': omni['flow_pressure'][trough_data['time']].values,
        'f107': omni['f107'][trough_data['time']].values,
        'dst': omni['dst'][trough_data['time']].values,
        'density': omni['proton_density'][trough_data['time']].values,
        'temp': omni['proton_temp'][trough_data['time']].values,
        'e_field': omni['e_field'][trough_data['time']].values,
    }
    bmag_mask = (params['bmag'] >= 4) & (params['bmag'] <= 5)
    kp_mask = params['kp'] < 3
    dst_mask = params['dst'] > 0
    # log
    for p, v in params.copy().items():
        if p in LOG:
            params[f"log_{p}"] = np.log10(v + 1)
    # Dither
    for p, v in params.items():
        if p in DITHER:
            params[p] = v + np.random.randn(*v.shape) * DITHER[p]

    clock_angle = np.arctan2(params['by'], params['bz'])
    clock_angle[clock_angle < 0] += 2 * np.pi
    params['clock_angle'] = clock_angle

    idx = np.argwhere(bmag_mask)[:, 0]
    idx = idx[idx < trough.shape[0] - 1]
    # plot_param_min_mlat(params['clock_angle'][idx], min_mlat[idx + 1], 0, 2, name='clock angle', lr=False)
    # plot_param_min_mlat(params['kp'], min_mlat, 0, 2, name='Kp', lr=True)
    # plot_param_min_mlat(params['kp'], min_mlat, 0, 5, name='Kp', lr=True)
    # plt.show()
    # plot_polar_probability_season(trough_data['time'][kp_mask], trough[kp_mask], np.isfinite(x[kp_mask]), "C:\\Users\\Greg\\Desktop\\study plots\\replication")
    plot_average_param_season(trough_data['time'][kp_mask], np.broadcast_to(config.mlt_vals[None, :], min_mlat.shape)[kp_mask], min_mlat[kp_mask], 30, MLT_BOUNDS, 'mlt', 'min_mlat', "C:\\Users\\Greg\\Desktop\\study plots\\replication")

    # Plot Enable Switches
    PARAM_DIST = False
    WIDTH_DEPTH = False
    WIDTH_PARAM = False
    DEPTH_PARAM = False
    PARAM_MLT = False
    PARAM_MLAT = False
    ####################################################################################################################
    # PARAM DIST #######################################################################################################
    ####################################################################################################################
    print("Parameter Distributions")
    if PARAM_DIST:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_param_hist(param_vals, 25, name=param_name, save_dir="C:\\Users\\Greg\\Desktop\\study plots\\param")
        plt.close('all')


    ####################################################################################################################
    # WIDTH - DEPTH ####################################################################################################
    ####################################################################################################################
    print("Width - Depth Distributions")
    if WIDTH_DEPTH:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_width_depth_set(params['width'], params['depth'], param_vals, param_name)
        plt.close('all')

    ####################################################################################################################
    # WIDTH - PARAM - MLT ##############################################################################################
    ####################################################################################################################
    print("Width - Parameter Distributions")
    if WIDTH_PARAM:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_param_mlt_set(params['width'], param_vals, "C:\\Users\\Greg\\Desktop\\study plots\\mlt_width", 'width', param_name, BINS['width'], WIDTH_BOUNDS)
        plt.close('all')

    ####################################################################################################################
    # DEPTH - PARAM - MLT ##############################################################################################
    ####################################################################################################################
    print("Depth - Parameter Distributions")
    if DEPTH_PARAM:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_param_mlt_set(params['depth'], param_vals, "C:\\Users\\Greg\\Desktop\\study plots\\mlt_depth", 'depth', param_name, BINS['depth'], DEPTH_BOUNDS)
        plt.close('all')

    ####################################################################################################################
    # PARAM - MLT ######################################################################################################
    ####################################################################################################################
    print("Parameter - MLT Distributions")
    if PARAM_MLT:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_param_mlt(trough, param_vals, np.isfinite(x), 25, name=param_name, save_dir="C:\\Users\\Greg\\Desktop\\study plots\\mlt")
                plot_param_mlt(trough[kp_mask], param_vals[kp_mask], np.isfinite(x[kp_mask]), 25, name=param_name, save_dir="C:\\Users\\Greg\\Desktop\\study plots\\mlt_low_kp")
                plot_param_mlt(trough[bmag_mask], param_vals[bmag_mask], np.isfinite(x[bmag_mask]), 25, name=param_name, save_dir="C:\\Users\\Greg\\Desktop\\study plots\\mlt_limited_bmag")
        plt.close('all')

    ####################################################################################################################
    # PARAM - MLAT #####################################################################################################
    ####################################################################################################################
    print("Parameter - MLAT Distributions")
    if PARAM_MLAT:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                for mlt in [-6, -3, 0, 3, 6]:
                    plot_param_mlat(trough, param_vals, np.isfinite(x), param_bins=25, name=param_name, mlt_center=mlt, save_dir="C:\\Users\\Greg\\Desktop\\study plots\\mlat")
                    plot_param_mlat(trough[kp_mask], param_vals[kp_mask], np.isfinite(x[kp_mask]), param_bins=25, name=param_name, mlt_center=mlt, save_dir="C:\\Users\\Greg\\Desktop\\study plots\\mlat_low_kp")
                    plot_param_mlat(trough[bmag_mask], param_vals[bmag_mask], np.isfinite(x[bmag_mask]), param_bins=25, name=param_name, mlt_center=mlt, save_dir="C:\\Users\\Greg\\Desktop\\study plots\\mlat_limited_bmag")
        plt.close('all')
    # plt.show()
