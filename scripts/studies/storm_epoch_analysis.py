import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
for i in range(len(zerox) - 1):
    interval = dst_i[zerox[i]:zerox[i + 1]]
    if np.all(interval > -50):
        continue
    idx = np.argmin(interval)
    mins.append(zerox[i] + idx)
mins = np.array(mins)
print()