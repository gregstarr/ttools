"""
Create trough dataset:
    0: No trough
    1: Non-SAPS trough
    2: SAPS trough
    3: unknown trough

Create plots:
    (low, medium, high Kp) x (SAPS, no SAPS)
"""
import numpy as np
import matplotlib.pyplot as plt
import apexpy
from scipy.stats import binned_statistic_dd
import bottleneck as bn
import pandas

from ttools import io, plotting, utils, config, satellite, tec
from ttools.old import get_s_data


def swarm_get(start_date, end_date):
    a, swarm_time = get_s_data(start_date, end_date, 'A')
    b, swarm_time = get_s_data(start_date, end_date, 'B')
    c, swarm_time = get_s_data(start_date, end_date, 'C')
    return {'A': a, 'B': b, 'C': c}, swarm_time


trough_data = np.load("C:\\Users\\Greg\\data\\dataset.npz")
trough = trough_data['trough'][36240:36270]
tec_times = trough_data['time'][36240:36270]

arb, _ = io.get_arb_data(tec_times[0], tec_times[-1] + np.timedelta64(1, 'h'))
dmsp_segments = satellite.get_segments_data(tec_times, 'dmsp')
swarm_segments = satellite.get_segments_data(tec_times, 'swarm', 'apex_lat', 'mlt', 'T_elec', get_data_func=swarm_get)

bins = [np.arange(tec_times[0], tec_times[-1] + np.timedelta64(2, 'h'), np.timedelta64(1, 'h')), np.arange(29.5, 90), np.arange(-12, 12 + 24 / 360, 48 / 360)]
superdarn = pandas.read_csv("C:\\Users\\Greg\\Downloads\\Starr,Gregory (1)\\Starr,Gregory\\20140219_north.csv", skiprows=14)
sdtime = pandas.to_datetime(superdarn['time']).values.astype('datetime64[s]')
apex = apexpy.Apex()
mlat, mlt = apex.convert(superdarn['vector_mlat'].values, superdarn['vector_mlon'].values, 'apex', 'mlt', datetime=sdtime)
mlt[mlt > 12] -= 24
theta = np.pi + np.pi * (mlt - 6) / 12 - np.deg2rad(superdarn['vector_kvect'])
fx = np.cos(theta) * superdarn['vector_vel_median'].values
fy = np.sin(theta) * superdarn['vector_vel_median'].values
sample = np.column_stack((sdtime.astype(int), mlat, mlt))
vx = binned_statistic_dd(sample, fx, 'median', bins).statistic
vy = binned_statistic_dd(sample, fy, 'median', bins).statistic
v = np.hypot(vx, vy)

saps = v > 400
saps = tec.remove_auroral(saps, arb)

fig, ax = plt.subplots(subplot_kw={'polar': True})
pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, trough[0] + 2 * saps[0], cmap='tab10')
ax.quiver((config.mlt_grid - 6) * np.pi / 12, 90 - config.mlat_grid, vx[0], vy[0], scale=30000, width=.001, cmap='jet')
fig.colorbar(pcm)
plotting.format_polar_mag_ax(ax)
fig.axes[-1].set_yticklabels(['none', '', 'trough', '', 'saps', '', 'saps + trough'])
plt.show()

for i in range(20, 22):
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    fig2, ax2 = plt.subplots(4, 2, tight_layout=True, sharex=True)
    fig3, ax3 = plt.subplots(3, 2, tight_layout=True, sharex=True)
    plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, trough[i], cmap='Blues')
    r = 90 - config.mlat_grid
    t = (config.mlt_grid - 6) * np.pi / 12
    ax.quiver(t - np.pi / 180, r + .5, vx[i], vy[i], np.hypot(vx[i], vy[i]), scale=40000, cmap='jet')

    for j1, (sat, sat_segments) in enumerate(dmsp_segments.items()):
        for j2, (direction, segments) in enumerate(sat_segments.items()):
            if len(segments) == 0:
                continue
            ax.scatter(np.pi * (segments[i]['mlt'] - 6) / 12, 90 - segments[i]['mlat'], s=2, c=segments[i]['hor_ion_v'], cmap='coolwarm', vmin=-1000, vmax=1000)
            ax.text(np.pi * (segments[i]['mlt'][0] - 6) / 12, 90 - segments[i]['mlat'][0], sat)
            ax2[j1, j2].plot(segments[i]['mlat'], segments[i]['hor_ion_v'])
            ax2[j1, j2].grid()
        ax2[j1, 0].set_ylabel(sat)

    for j1, (sat, sat_segments) in enumerate(swarm_segments.items()):
        for j2, (direction, segments) in enumerate(sat_segments.items()):
            if len(segments) == 0:
                continue
            ax.scatter(np.pi * (segments[i]['mlt'] - 6) / 12, 90 - segments[i]['apex_lat'], s=2, c=segments[i]['T_elec'], cmap='jet', vmin=0, vmax=4000)
            ax.text(np.pi * (segments[i]['mlt'][0] - 6) / 12, 90 - segments[i]['apex_lat'][0], sat)
            ax3[j1, j2].plot(segments[i]['apex_lat'], segments[i]['T_elec'], 'r')
            ax3[j1, j2].grid()
            ax3[j1, j2].twinx().plot(segments[i]['apex_lat'], segments[i]['smooth_dne'], 'g')
        ax3[j1, 0].set_ylabel(sat)

    plotting.format_polar_mag_ax(ax)

plt.show()
