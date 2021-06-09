import numpy as np
import bottleneck as bn
import pandas
import os
import matplotlib.pyplot as plt
from matplotlib import colors

from ttools import config, io, plotting


MLT_DITHER = .01
MLAT_DITHER = .01
KP_DITHER = .5
LOG_KP_DITHER = .1
E_FIELD_DITHER = .01

MLT_BINS = 40
MLAT_BINS = 40
CSA_BINS = 40

MLT_BOUNDS = [-12, 12]
MLAT_BOUNDS = [40, 80]
CSA_BOUNDS = [-4, 0]


def plot_param_mlt(trough, param, param_bins=50, param_bounds=None, name='param', norm=None, save_dir=None, time_mask=None, file_extra=None, title_extra=None):
    if time_mask is None:
        time_mask = np.ones(trough.shape[0], dtype=bool)

    mask = np.any(trough[time_mask], axis=1)
    x = np.broadcast_to(config.mlt_vals[None, :], mask.shape)
    x = x + np.random.randn(*x.shape) * MLT_DITHER

    y_sl = (time_mask, ) + (None, ) * (2 - param.ndim)
    y = np.broadcast_to(param[y_sl], mask.shape)

    if param_bounds is None:
        param_bounds = np.quantile(param[np.isfinite(param)], [.01, .99])

    mask &= np.isfinite(y)

    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    (counts, *_, pcm) = ax.hist2d(x[mask], y[mask], bins=[MLT_BINS, param_bins], range=[MLT_BOUNDS, param_bounds], cmap='jet', norm=norm)
    plt.colorbar(pcm)
    title = f"N = {counts.sum()}{' ' + title_extra if title_extra is not None else ''}"
    ax.set_title(title)
    ax.set_xlabel('MLT')
    ax.set_ylabel(name)

    if save_dir is not None:
        fn = f"{name}_mlt_dist{'_' + file_extra if file_extra is not None else ''}{'_norm' if norm is not None else ''}.png"
        fig.savefig(os.path.join(save_dir, fn))


def plot_param_mlt_set(param, set_param, save_dir, name='param', set_name='set_param', bins=50, param_bounds=None, quantiles=(0, .2, .4, .6, .8, 1)):
    edges = np.quantile(set_param[np.isfinite(set_param)], quantiles)
    bounds = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    for i, bound in enumerate(bounds):
        time_mask = (set_param >= bound[0]) & (set_param <= bound[1])
        title_extra = f"|| {set_name} = ({bound[0]:.2f}, {bound[1]:.2f})"
        plot_param_mlt(trough, param, bins, param_bounds, name, time_mask=time_mask, title_extra=title_extra,
                       file_extra=f"{set_name}_{i}", save_dir=save_dir)


def plot_lparam_tparam(l_param, t_param, lparam_bins=50, tparam_bins=50, tname='tparam', lname='lparam', mlt_center=0, mlt_width=1.5, save_dir=None):
    mlt_mask = abs(config.mlt_vals - mlt_center) <= mlt_width
    x = np.broadcast_to(t_param[:, None], l_param.shape)

    tparam_bounds = np.nanquantile(t_param, [.01, .99])
    lparam_bounds = np.nanquantile(l_param[l_param != 0], [.01, .99])

    mask = np.isfinite(t_param)[:, None] & (l_param != 0) & mlt_mask[None, :]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
    counts, *_, pcm = ax[0].hist2d(x[mask], l_param[mask], bins=[tparam_bins, lparam_bins], range=[tparam_bounds, lparam_bounds], cmap='jet')
    plt.colorbar(pcm)
    pcm = ax[1].pcolormesh(counts.T / np.sum(counts.T, axis=0, keepdims=True), cmap='jet')
    plt.colorbar(pcm)
    ax[0].set_title(f"N = {counts.sum()} || MLT = {mlt_center}")
    ax[0].set_xlabel(tname)
    ax[0].set_ylabel(lname)
    if save_dir is not None:
        fn = f"{tname}_{lname}_dist{mlt_center % 24:d}.png"
        fig.savefig(os.path.join(save_dir, fn))


if __name__ == "__main__":
    # Load trough dataset
    trough_data = np.load("E:\\dataset.npz")
    trough = trough_data['trough']
    x = trough_data['x']
    csa = np.nansum(x * trough, axis=1)
    # Load Omni
    omni = io.get_omni_data()
    # Assemble
    kp = io.get_kp(trough_data['time'])
    log_kp = np.log10(kp + 1)
    e_field = omni['e_field'][trough_data['time']].values
    plot_lparam_tparam(csa, e_field, 40, 40, 'e_field', 'csa', -6)
    # plot_param_mlt_set(csa, e_field, "E:\\study plots\\mlt_csa", 'csa', 'csa', CSA_BINS, CSA_BOUNDS)
    plt.show()
