"""
- trough distribution
- total trough size centered at onset
- for any longitudinal parameter: min mlat, width, depth
    center each example at onset, look at time series before and after
- "center of mass"
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from ttools import io, config

plt.style.use('ggplot')


def sea(t0, f, dt=12):
    idx = t0[:, None] + np.arange(-dt, dt + 1)
    f_series = f[idx]
    f_series = f_series - np.nanmean(f_series, axis=1, keepdims=True)
    # f_series = f_series - np.nanmean(f_series[:, [0, 1, 2]], axis=1, keepdims=True)
    return f_series


trough_data = np.load("E:\\dataset.npz")
trough = trough_data['trough']
X = -trough_data['x']

# Calculate trough depth / width
X[~trough] = -np.inf
min_mlat = config.mlat_vals[np.nanargmax(X, axis=1)]
min_mlat[~np.any(trough, axis=1)] = np.nan
width = np.sum(trough, axis=1).astype(float)
width[width == 0] = np.nan
X[~trough] = np.nan
depth = np.nanmax(X, axis=1)

# Load Omni
omni = io.get_omni_data()
dst = omni['dst'][trough_data['time']].values
dst_i = dst.copy()
fin_mask = np.isfinite(dst)
ut = trough_data['time'].astype(int)
interpolator = interp1d(ut[fin_mask], dst[fin_mask], kind='previous', bounds_error=False, fill_value=0)
dst_i[~fin_mask] = interpolator(ut[~fin_mask])
zerox, = np.nonzero(np.diff(dst_i <= -50))
zerox += 1
zerox = np.concatenate(([0], zerox, [dst_i.shape[0] - 1]))

mins = []
min_dst = []
for i in range(len(zerox) - 1):
    interval = dst_i[zerox[i]:zerox[i + 1]]
    if np.all(interval > -50):
        continue
    idx = np.argmin(interval)
    min_dst.append(np.min(interval))
    mins.append(zerox[i] + idx)
mins = np.array(mins)
min_dst = np.array(min_dst)

min_mlat_sea = sea(mins, min_mlat)
t = np.arange(-12, 13)

color_val = (min_dst + 50) / np.min(min_dst + 50)
colors = plt.cm.jet(color_val)
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(len(mins)):
    print(i)
    ax.plot(t, np.nanmean(min_mlat_sea[i], axis=1), color=colors[i], lw=color_val[i]+.1, alpha=.5)

#
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(t, np.nanmean(min_mlat_sea, axis=(0, 2)))
# ax[0].plot(t, np.nanmedian(min_mlat_sea, axis=(0, 2)))
# ax[0].plot(t, np.nanquantile(min_mlat_sea, .25, axis=(0, 2)), 'k--')
# ax[0].plot(t, np.nanquantile(min_mlat_sea, .75, axis=(0, 2)), 'k--')
# ax[0].set_ylabel('mlat')
#
# min_mlat_sea = min_mlat_sea + np.random.randn(*min_mlat_sea.shape) * .2
# t = np.broadcast_to(t[None, :, None], min_mlat_sea.shape)
# mask = np.isfinite(min_mlat_sea)
#
# ax[1].hist2d(t[mask], min_mlat_sea[mask], bins=[25, 100], range=[[-12, 12], [-10, 10]])

plt.show()
