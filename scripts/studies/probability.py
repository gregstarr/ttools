import numpy as np
import bottleneck as bn
from scipy import interpolate
import matplotlib.pyplot as plt

from ttools import config, io, plotting, convert, utils


MONTH_INDEX = {
    'winter': [0, 1, 10, 11],
    'equinox': [2, 3, 8, 9],
    'summer': [4, 5, 6, 7],
}


def plot_mlt_mlat_polar_probability(trough_data, kp):
    months = (trough_data['time'].astype('datetime64[M]') - trough_data['time'].astype('datetime64[Y]')).astype(int)
    kp_mask = kp <= 3

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


def plot_season_mlt_probability(trough_data, kp):
    kp_mask = kp <= 3

    n_season_bins = 50
    n_mlt_bins = 60
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


def plot_kp_mlat_probability(trough_data, kp):
    kp_bins = np.linspace(0, 8, 17)
    kp_vals = np.column_stack((kp_bins[:-1], kp_bins[1:])).mean(axis=1)
    mlt_centers = np.array([-3, 0, 3])
    kp_grid, mlat_grid = np.meshgrid(kp_vals, config.mlat_vals)
    prob = np.empty((mlt_centers.shape[0], kp_grid.shape[0], kp_grid.shape[1]))

    for m, mlt_center in enumerate(mlt_centers):
        mlt_mask = abs(config.mlt_vals - mlt_center) <= 1.5
        for i in range(kp_vals.shape[0]):
            mask = (kp >= kp_bins[i]) & (kp < kp_bins[i + 1])
            prob[m, :, i] = np.mean(np.any(trough_data['trough'][mask][:, :, mlt_mask], axis=2), axis=0)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    for m, mlt_center in enumerate(mlt_centers):
        pcm = ax[m].pcolormesh(kp_grid, mlat_grid, prob[m], cmap='jet', vmin=0, vmax=prob.max())
        ax[m].set_title(mlt_center)
    ax[0].set_ylabel("MLAT")
    ax[1].set_xlabel("Kp")

    fig, cax = plt.subplots()
    plt.colorbar(pcm, cax=cax)


def plot_kp_mlt_probability(trough_data, kp):
    kp_bins = np.linspace(0, 8, 17)
    kp_vals = np.column_stack((kp_bins[:-1], kp_bins[1:])).mean(axis=1)
    n_mlt_bins = 30
    mlt_vals = np.linspace(-12, 12, n_mlt_bins)
    n_mlt = trough_data['trough'].shape[-1]
    kp_grid, mlt_grid = np.meshgrid(kp_vals, mlt_vals)
    prob = np.empty((kp_grid.shape[0], kp_grid.shape[1]))

    for i in range(kp_vals.shape[0]):
        mask = (kp >= kp_bins[i]) & (kp < kp_bins[i + 1])
        prob[:, i] = np.mean(np.any(trough_data['trough'][mask][:, :, np.arange(n_mlt).reshape((-1, n_mlt // n_mlt_bins))], axis=(1, 3)), axis=0)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(kp_grid, mlt_grid, prob, cmap='jet')
    plt.colorbar(pcm)
    ax.set_ylabel("MLT")
    ax.set_xlabel("Kp")


def plot_param_mlat_probability(trough_data, param, nbins, pmin=None, pmax=None, name='param', mlt_centers=np.array([-3, 0, 3])):
    if pmin is None:
        pmin = param.min()
    if pmax is None:
        pmax = param.max()
    param_bins = np.linspace(pmin, pmax, nbins + 1)
    param_vals = np.column_stack((param_bins[:-1], param_bins[1:])).mean(axis=1)
    param_grid, mlat_grid = np.meshgrid(param_vals, config.mlat_vals)
    prob = np.empty((mlt_centers.shape[0], param_grid.shape[0], param_grid.shape[1]))

    for m, mlt_center in enumerate(mlt_centers):
        mlt_mask = abs(config.mlt_vals - mlt_center) <= 1.5
        for i in range(param_vals.shape[0]):
            mask = (param >= param_bins[i]) & (param < param_bins[i + 1])
            prob[m, :, i] = np.mean(np.any(trough_data['trough'][mask][:, :, mlt_mask], axis=2), axis=0)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, tight_layout=True)
    for m, mlt_center in enumerate(mlt_centers):
        pcm = ax[m].pcolormesh(param_grid, mlat_grid, prob[m], cmap='jet', vmin=0, vmax=np.nanmax(prob))
        ax[m].set_title(mlt_center)
    ax[0].set_ylabel("MLAT")
    ax[1].set_xlabel(name)
    plt.colorbar(pcm)


if __name__ == "__main__":
    trough_data_fn = "E:\\labels.npz"
    trough_data = np.load(trough_data_fn)
    kp = io.get_kp(trough_data['time'])
    omni = io.get_omni_data()

    speed = omni['plasma_speed'][trough_data['time']].interpolate()
    bmag = omni['b_mag'][trough_data['time']].interpolate()

    # plot_mlt_mlat_polar_probability(trough_data, kp)
    # plot_season_mlt_probability(trough_data, kp)
    # plot_kp_mlt_probability(trough_data, kp)
    # plot_param_mlat_probability(trough_data, kp, 16, 'Kp')
    plot_param_mlat_probability(trough_data, speed, 20, pmax=800, name='Speed')
    plot_param_mlat_probability(trough_data, bmag, 10, pmax=30, name='BMag')

    plt.show()
