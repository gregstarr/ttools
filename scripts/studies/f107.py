import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from ttools import io


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
f107 = omni['f107'][trough_data['time']].values
f107 += np.random.randn(*f107.shape) * 3
mask = np.isfinite(depth) * np.isfinite(f107)[:, None]
f107_bounds = np.quantile(f107[np.isfinite(f107)], [.01, .99])
depth_bounds = np.quantile(depth[np.isfinite(depth)], [.01, .99])
plt.hist2d(np.broadcast_to(f107[:, None], depth.shape)[mask], depth[mask], bins=40, range=[f107_bounds, depth_bounds], norm=colors.LogNorm())
plt.show()
