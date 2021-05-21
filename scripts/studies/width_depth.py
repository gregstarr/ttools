import numpy as np
import matplotlib.pyplot as plt

from ttools import io, config, swarm, plotting


if __name__ == "__main__":
    trough_data_fn = "E:\\dataset.npz"
    trough_data = np.load(trough_data_fn)
    trough = trough_data['trough']
    x = trough_data['x']
    xi = x.copy()
    xi[~trough] = np.inf
    kp = io.get_kp(trough_data['time'])
    tec, *_ = io.get_tec_data(trough_data['time'][0], trough_data['time'][-1] + np.timedelta64(1, 'h'))
    omni = io.get_omni_data()

    mlt_mask = (config.mlt_vals >= 3) & (config.mlt_vals <= 6)
    depth = np.nanmin(xi[kp <= 2][:, :, mlt_mask], axis=1)
    depth[depth == np.inf] = np.nan
    width = trough[kp <= 2][:, :, mlt_mask].sum(axis=1).astype(float)
    width[width == 0] = np.nan

    idx = np.unique(np.argwhere(np.sum((depth < -3), axis=1) >= 1)[:, 0])

    for i in idx[::-1]:
        print(i)
        swarm_segments = swarm.get_segments_data(trough_data['time'][kp <= 2][i:i+1])
        swarm_troughs = swarm.get_swarm_troughs(swarm_segments)

        polar_fig, polar_ax = plt.subplots(1, 3, figsize=(18, 10), subplot_kw=dict(projection='polar'), tight_layout=True)
        line_fig, line_ax = plt.subplots(3, 2, figsize=(20, 12), tight_layout=True, sharex=True, sharey=True)
        plotting.polar_pcolormesh(polar_ax[0], config.mlat_grid, config.mlt_grid, tec[kp <= 2][i], vmin=0, vmax=20)
        plotting.polar_pcolormesh(polar_ax[1], config.mlat_grid, config.mlt_grid, x[kp <= 2][i], vmin=-.5, vmax=.5, cmap='coolwarm')
        plotting.polar_pcolormesh(polar_ax[2], config.mlat_grid, config.mlt_grid, trough[kp <= 2][i], vmin=0, vmax=1, cmap='Blues')
        plotting.format_polar_mag_ax(polar_ax)

        for j1, sat in enumerate(['A', 'B', 'C']):
            for j2, d in enumerate(['up', 'down']):
                try:
                    line_ax[j1, j2].plot(swarm_segments[sat][d][0]['mlat'], swarm_segments[sat][d][0]['smooth_dne'])
                    line_ax[j1, j2].set_title(f"mlt: {swarm_segments[sat][d][0]['mlt'][0]}, sat: {sat}")
                    line_ax[j1, j2].grid(True)
                    polar_ax[0].plot(np.pi * (swarm_segments[sat][d][0]['mlt'] - 6) / 12, 90 - swarm_segments[sat][d][0]['mlat'], 'k--')
                    polar_ax[1].plot(np.pi * (swarm_segments[sat][d][0]['mlt'] - 6) / 12, 90 - swarm_segments[sat][d][0]['mlat'], 'k--')
                    polar_ax[2].plot(np.pi * (swarm_segments[sat][d][0]['mlt'] - 6) / 12, 90 - swarm_segments[sat][d][0]['mlat'], 'k--')
                except:
                    pass

        param_fig, param_ax = plt.subplots(3, 1, figsize=(20, 12), tight_layout=True, sharex=True)
        param_ax[0].plot(trough_data['time'][kp <= 2][i - 24:i + 1], omni['plasma_speed'][trough_data['time'][kp <= 2][i - 24:i + 1]])
        param_ax[0].grid(True)
        param_ax[0].set_ylabel('Vsw')
        param_ax[1].plot(trough_data['time'][kp <= 2][i - 24:i + 1], omni['bz_gsm'][trough_data['time'][kp <= 2][i - 24:i + 1]])
        param_ax[1].grid(True)
        param_ax[1].set_ylabel('Bz')
        param_ax[2].plot(trough_data['time'][kp <= 2][i - 24:i + 1], kp[kp <= 2][i - 24:i + 1])
        param_ax[2].grid(True)
        param_ax[2].set_ylabel('Kp')

        polar_fig.savefig(f"E:\\temp\\{i}_polar.png")
        line_fig.savefig(f"E:\\temp\\{i}_line.png")
        param_fig.savefig(f"E:\\temp\\{i}_params.png")
        plt.close(polar_fig)
        plt.close(line_fig)
        plt.close(param_fig)
