import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from ttools import io, config

plt.style.use('ggplot')


trough_data = np.load("C:\\Users\\Greg\\data\\dataset.npz")
trough = trough_data['trough']
X = -1 * trough_data['x']
X[~trough] = np.nan
kp = np.nanmean(io.get_kp(trough_data['time']).reshape((-1, 24)), axis=1)

theta = (config.mlt_vals - 6) * np.pi / 12
r = 90 - config.mlat_vals
x_grid = np.cos(theta)[None, :] * r[:, None]
y_grid = np.sin(theta)[None, :] * r[:, None]

fig1, ax1 = plt.subplots(1, 2)
fig2, ax2 = plt.subplots(2, 2, sharex=True)
a = 60 * 180
# area based
t_x = np.sum(np.broadcast_to(x_grid, trough.shape) * trough, axis=(1, 2)) / a
t_y = np.sum(np.broadcast_to(y_grid, trough.shape) * trough, axis=(1, 2)) / a
t_x = t_x.reshape((-1, 24)).sum(axis=1)
t_y = t_y.reshape((-1, 24)).sum(axis=1)
mask = (abs(t_x) > 0) & (abs(t_y) > 0) & np.isfinite(kp)
ax1[0].scatter(t_x[mask], t_y[mask], s=2, c=kp[mask])
ax2[0, 0].plot(kp[mask], t_x[mask], '.', ms=2)
ax2[1, 0].plot(kp[mask], t_y[mask], '.', ms=2)
ax2[1, 0].set_xlabel('Kp')

# volume based
t_x = np.nansum(np.broadcast_to(x_grid, X.shape) * X, axis=(1, 2)) / a
t_y = np.nansum(np.broadcast_to(y_grid, X.shape) * X, axis=(1, 2)) / a
t_x = t_x.reshape((-1, 24)).sum(axis=1)
t_y = t_y.reshape((-1, 24)).sum(axis=1)
mask = (abs(t_x) > 0) & (abs(t_y) > 0) & np.isfinite(kp)
ax1[1].scatter(t_x[mask], t_y[mask], s=2, c=kp[mask])
ax2[0, 1].plot(kp[mask], t_x[mask], '.', ms=2)
ax2[1, 1].plot(kp[mask], t_y[mask], '.', ms=2)

plt.show()
