import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas

from ttools import config, io

MLT_DITHER = .025
MLT_BOUNDS = (-12, 12)
MLT_BINS = 40

KP_DITHER = 0
DST_DITHER = 0
NEWELL_DITHER = 0
AE_DITHER = 0
AP_DITHER = 0


def plot_param_min_mlat(param, mlat_min, center, width, bounds=None, nbins=20, name='param', lr=True):
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
    ax.set_ylabel('MLT of Trough Minimum')
    ax.grid()


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
    # kp = io.get_kp(trough_data['time'])
    # kp += np.random.randn(*kp.shape) * KP_DITHER
    # newell = pandas.read_hdf("E:\\newell.h5").values
    # newell += np.random.randn(*newell.shape) * NEWELL_DITHER
    # dst = omni['dst'][trough_data['time']].values
    # dst += np.random.randn(*dst.shape) * DST_DITHER
    # dst_mask = dst < 5
    # ap = omni['ap'][trough_data['time']].values
    # ap += np.random.randn(*ap.shape) * AP_DITHER

    sme = pandas.read_hdf("E:\\sme.h5").values
    ae = omni['ae'][trough_data['time']].values

    plot_param_min_mlat(sme, min_mlat, 0, 2, name='sme', lr=True)
    plot_param_min_mlat(ae, min_mlat, 0, 2, name='ae', lr=True)
    plt.show()
