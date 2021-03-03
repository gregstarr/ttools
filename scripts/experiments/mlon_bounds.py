import numpy as np
import matplotlib.pyplot as plt

from ttools import io, config

plt.style.use('ggplot')

start_date = np.datetime64("2010-01")
end_date = np.datetime64("2020-01")
dt = np.timedelta64(1, 'M')
mlt = config.mlt_grid
bins = np.arange(-1, 360, 2)

hist_totals = np.zeros(bins.shape[0] - 1)

while start_date + dt < end_date:
    tec, ref_times, ssmlon, n_samples = io.get_tec_data(start_date, start_date + dt)
    mlon = (15 * mlt[None, :, :] - 180 + ssmlon[:, None, None] + 360) % 360
    mlon[mlon > bins.max()] -= 360
    hist, _ = np.histogram(mlon[np.isfinite(tec)], bins=bins)
    hist_totals += hist
    start_date += dt

plt.plot(hist_totals)
plt.show()
