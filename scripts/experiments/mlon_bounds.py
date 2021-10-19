import numpy as np
import matplotlib.pyplot as plt

from ttools import io, config

plt.style.use('ggplot')

start_date = np.datetime64("2010-01")
date = start_date
end_date = np.datetime64("2020-01")
dt = np.timedelta64(6, 'M')
mlt = config.mlt_grid
bins = np.arange(-1, 360, 2)

hist_totals = np.zeros(bins.shape[0] - 1)
coverage = []
N = 0

while date < end_date:
    tec, ref_times, ssmlon, n_samples = io.get_tec_data(date, date + dt)
    N += tec.shape[0] * tec.shape[1]
    coverage.append(np.isfinite(tec).mean())
    mlon = (15 * mlt[None, :, :] - 180 + ssmlon[:, None, None] + 360) % 360
    mlon[mlon > bins.max()] -= 360
    hist, _ = np.histogram(mlon[np.isfinite(tec)], bins=bins)
    hist_totals += hist
    date += dt

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].plot(np.arange(0, 360, 2), hist_totals / N)
ax[0].set_title('Coverage over MLON')
ax[0].set_xlabel('MLON')
ax[0].set_ylabel('Data Coverage Proportion')
ax[1].plot(np.arange(start_date, end_date, dt), coverage)
ax[1].set_title('Coverage over time')
ax[1].set_xlabel('Year')
plt.show()
