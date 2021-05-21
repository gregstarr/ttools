import numpy as np
import bottleneck as bn
import pandas
import os
import matplotlib.pyplot as plt
from matplotlib import colors

from ttools import config, io, plotting


MONTH_INDEX = {
    'winter': [0, 1, 10, 11],
    'equinox': [2, 3, 8, 9],
    'summer': [4, 5, 6, 7],
}

DITHER = {
    'mlt': .01,
    'mlat': .01,
    'width': .7,
    'kp': .5,
    'log_kp': .1,
    'by': .07,
    'bz': .07,
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


def plot_polar_probability_season(trough_data, kp_mask):
    months = (trough_data['time'].astype('datetime64[M]') - trough_data['time'].astype('datetime64[Y]')).astype(int)

    prob = {}
    for season, mo in MONTH_INDEX.items():
        mask = np.zeros_like(months, dtype=bool)
        for m in mo:
            mask |= months == m
        mask &= kp_mask
        prob[season] = bn.nanmean(trough_data['trough'][mask], 0)
    max_prob = max([p.max() for p in prob.values()])

    fig = plt.figure(figsize=(16, 6), tight_layout=True)
    gs = plt.GridSpec(1, 4, width_ratios=[20, 20, 20, 1])

    for i, (season, p) in enumerate(prob.items()):
        ax = fig.add_subplot(gs[i], polar=True)
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, p, cmap='jet', vmin=0, vmax=max_prob)
        ax.set_title(season, loc='left')
        plotting.format_polar_mag_ax(ax, tick_color='grey')

    cb_ax = fig.add_subplot(gs[3])
    plt.colorbar(pcm, cax=cb_ax)


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


def plot_season_mlt_probability(trough_data, kp_mask, n_season_bins=50, n_mlt_bins=60):
    n_mlt = trough_data['trough'].shape[-1]

    seconds = (trough_data['time'].astype('datetime64[s]') - trough_data['time'].astype('datetime64[Y]')).astype('timedelta64[s]').astype(float)
    bin_edges = np.linspace(0, 60 * 60 * 24 * 365, n_season_bins + 1)

    season, mlt = np.meshgrid(np.linspace(0, 12, n_season_bins), np.linspace(-12, 12, n_mlt_bins))
    prob = np.empty((n_mlt_bins, n_season_bins))
    for i in range(n_season_bins):
        mask = (seconds >= bin_edges[i]) & (seconds < bin_edges[i + 1]) & kp_mask
        prob[:, i] = np.mean(np.any(trough_data['trough'][mask][:, :, np.arange(n_mlt).reshape((-1, n_mlt // n_mlt_bins))], axis=(1, 3)), axis=0)

    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[20, 1]))
    pcm = ax.pcolormesh(season, mlt, prob, cmap='jet')
    plt.colorbar(pcm, cax=cax)
    ax.set_xticks(np.arange(.5, 12, 2))
    ax.set_xticklabels(['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
    ax.set_ylabel("MLT")


def plot_param_mlt(trough, param, param_bins=50, param_bounds=None, name='param', norm=None, save_dir=None, time_mask=None, file_extra=None, title_extra=None):
    if time_mask is None:
        time_mask = np.ones(trough.shape[0], dtype=bool)

    mask = np.any(trough[time_mask], axis=1)
    x = np.broadcast_to(config.mlt_vals[None, :], mask.shape)
    x = x + np.random.randn(*x.shape) * DITHER['mlt']

    y_sl = (time_mask, ) + (None, ) * (2 - param.ndim)
    y = np.broadcast_to(param[y_sl], mask.shape)

    if param_bounds is None:
        param_bounds = np.quantile(param[np.isfinite(param)], [.01, .99])

    mask &= np.isfinite(y)

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    (counts, *_, pcm) = ax.hist2d(x[mask], y[mask], bins=[BINS['mlt'], param_bins], range=[MLT_BOUNDS, param_bounds], cmap='jet', norm=norm)
    plt.colorbar(pcm)
    title = f"N = {counts.sum()}{' ' + title_extra if title_extra is not None else ''}"
    ax.set_title(title)
    ax.set_xlabel('MLT')
    ax.set_ylabel(name)

    if save_dir is not None:
        fn = f"{name}_mlt_dist{'_' + file_extra if file_extra is not None else ''}{'_norm' if norm is not None else ''}.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_param_mlat(trough, param, param_bins=50, param_bounds=None, name='param', norm=None, mlt_center=0, mlt_width=1.5, save_dir=None):
    shp = (trough.shape[0], trough.shape[1])
    x = np.broadcast_to(param[:, None], shp)
    y = np.broadcast_to(config.mlat_vals[None, :], shp)
    y = y + np.random.randn(*y.shape) * DITHER['mlat']

    if param_bounds is None:
        param_bounds = np.quantile(param[np.isfinite(param)], [.01, .99])

    mlt_mask = abs(config.mlt_vals - mlt_center) <= mlt_width
    trough_mask = np.any(trough[:, :, mlt_mask], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    (counts, *_, pcm) = ax.hist2d(x[trough_mask], y[trough_mask], bins=[param_bins, BINS['mlat']], range=[param_bounds, MLAT_BOUNDS], cmap='jet', norm=norm)
    plt.colorbar(pcm)
    ax.set_title(f"N = {counts.sum()} || MLT = {mlt_center}")
    ax.set_xlabel(name)
    ax.set_ylabel('MLAT')
    if save_dir is not None:
        fn = f"{name}_mlat_dist{mlt_center % 24:d}{'_norm' if norm is not None else ''}.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_width_depth(width, depth, time_mask, mlt_mask, file_extra=None, title_extra=None, save_dir=None, norm=None):
    x = width[time_mask][:, mlt_mask]
    y = depth[time_mask][:, mlt_mask]

    mask = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
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
            plot_width_depth(width, depth, time_mask, mlt_mask, file_extra, title_extra, "E:\\study plots\\width_depth")


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
    width = np.sum(trough, axis=1).astype(float)
    width[width == 0] = np.nan
    depth = np.nanmin(x, axis=1)
    depth[depth == np.inf] = np.nan
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
        'speed': omni['plasma_speed'][trough_data['time']].values,
        'pressure': omni['flow_pressure'][trough_data['time']].values,
        'f107': omni['f107'][trough_data['time']].values,
        'dst': omni['dst'][trough_data['time']].values,
        'density': omni['proton_density'][trough_data['time']].values,
        'temp': omni['proton_temp'][trough_data['time']].values,
    }
    # log
    for p, v in params.copy().items():
        if p in LOG:
            params[f"log_{p}"] = np.log10(v + 1)
    # Dither
    for p, v in params.items():
        if p in DITHER:
            params[p] = v + np.random.randn(*v.shape) * DITHER[p]

    # Plot Enable Switches
    PARAM_DIST = True
    WIDTH_DEPTH = True
    WIDTH_PARAM = True
    DEPTH_PARAM = True
    PARAM_MLT = True
    PARAM_MLAT = True
    ####################################################################################################################
    # PARAM DIST #######################################################################################################
    ####################################################################################################################
    print("Parameter Distributions")
    if PARAM_DIST:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_param_hist(param_vals, 100, name=param_name, save_dir="E:\\study plots\\param")

    ####################################################################################################################
    # WIDTH - DEPTH ####################################################################################################
    ####################################################################################################################
    print("Width - Depth Distributions")
    if WIDTH_DEPTH:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_width_depth_set(params['width'], params['depth'], param_vals, param_name)

    ####################################################################################################################
    # WIDTH - PARAM ####################################################################################################
    ####################################################################################################################
    print("Width - Parameter Distributions")
    if WIDTH_PARAM:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_param_mlt_set(params['width'], param_vals, "E:\\study plots\\mlt_width", 'width', param_name, BINS['width'], WIDTH_BOUNDS)

    ####################################################################################################################
    # DEPTH - PARAM ####################################################################################################
    ####################################################################################################################
    print("Depth - Parameter Distributions")
    if DEPTH_PARAM:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_param_mlt_set(params['depth'], param_vals, "E:\\study plots\\mlt_depth", 'depth', param_name, BINS['depth'], DEPTH_BOUNDS)

    ####################################################################################################################
    # PARAM - MLT ######################################################################################################
    ####################################################################################################################
    print("Parameter - MLT Distributions")
    if PARAM_MLT:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                plot_param_mlt(trough, param_vals, 100, name=param_name, save_dir="E:\\study plots\\mlt")
                plot_param_mlt(trough, param_vals, 100, name=param_name, norm=colors.LogNorm(), save_dir="E:\\study plots\\mlt")

    ####################################################################################################################
    # PARAM - MLAT #####################################################################################################
    ####################################################################################################################
    print("Parameter - MLAT Distributions")
    if PARAM_MLAT:
        for param_name, param_vals in params.items():
            if param_vals.ndim == 1:
                for mlt in [-6, -3, 0, 3, 6]:
                    plot_param_mlat(trough, param_vals, name=param_name, mlt_center=mlt, save_dir="E:\\study plots\\mlat")
                    plot_param_mlat(trough, param_vals, name=param_name, mlt_center=mlt, norm=colors.LogNorm(), save_dir="E:\\study plots\\mlat")

    # plt.show()
