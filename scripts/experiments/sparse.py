import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from skimage.util import view_as_windows

from ttools import io, config, utils

plt.style.use('ggplot')

# Load Omni
omni = io.get_omni_data()
bz = omni['bz_gsm'][400000:].values
L = 24
N = 3
p = np.random.randn((L, N))
arr = view_as_windows(bz, (L,), (1,))
mask = np.isfinite(arr)
for j in range(N):
    c = .5 / np.nanvar(arr, axis=1)
    c[~np.isfinite(c)] = 10
    for i in range(40):
        pb = np.broadcast_to(p[None, :, j], arr.shape)
        a = np.nansum(arr * pb, axis=1) / np.sum((pb * mask) ** 2, axis=1)
        loss = np.sum(-1 / (c * np.nansum((a[:, None] * p[None, :, j] - arr) ** 2, axis=1) + 1))
        grad = 2 * np.nansum(c[:, None] * a[:, None] * (a[:, None] * p[None, :, j] - arr) / ((c * np.nansum((a[:, None] * p[None, :, j] - arr) ** 2, axis=1) + 1) ** 2)[:, None], axis=0)
        p -= .1 * grad
        print(loss)
    m = 1 / (c * np.nansum((a[:, None] * p[None, :, j] - arr) ** 2, axis=1) + 1)
    print()


fig, ax = plt.subplots()
ax.plot(p)

plt.show()
