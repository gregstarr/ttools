if __name__ == "__main__":
    raise Exception("Broken")

    import matplotlib.pyplot as plt
    import numpy as np
    from ttools import swarm, utils, io, rbf_inversion, compare, plotting, config

    start_time = np.datetime64("2016-06-10T23:00:00")
    end_time = np.datetime64("2016-06-12T01:00:00")

    tec, times, ssmlon, n = io.get_tec_data(start_time, end_time)
    tec_troughs, x = rbf_inversion.get_tec_troughs(tec, times)
    tec, times, ssmlon = utils.moving_func_trim(3, tec, times, ssmlon)

    swarm_segments = swarm.get_segments_data(times)
    swarm_troughs = swarm.get_swarm_troughs(swarm_segments)

    results = compare.compare(times, tec_troughs, swarm_troughs, ssmlon)

    for t in range(times.shape[0]):
        fig, ax = plt.subplots(1, 3, figsize=(20, 12), subplot_kw=dict(projection='polar'), tight_layout=True)
        plotting.polar_pcolormesh(ax[0], config.mlat_grid, config.mlt_grid, tec[t], vmin=0, vmax=20)
        plotting.plot_swarm_troughs_polar(ax[0], swarm_troughs[swarm_troughs['tec_ind'] == t])
        plotting.polar_pcolormesh(ax[1], config.mlat_grid, config.mlt_grid, x[t], vmin=-.5, vmax=.5, cmap='coolwarm')
        plotting.plot_swarm_troughs_polar(ax[1], swarm_troughs[swarm_troughs['tec_ind'] == t])
        plotting.polar_pcolormesh(ax[2], config.mlat_grid, config.mlt_grid, tec_troughs[t], vmin=0, vmax=1, cmap='Blues')
        plotting.plot_swarm_troughs_polar(ax[2], swarm_troughs[swarm_troughs['tec_ind'] == t])
        plotting.format_polar_mag_ax(ax)
        fig.savefig(f"E:\\temp_plots\\{t}_polar.png")
        plt.close(fig)

        fig, ax = plt.subplots(3, 2, figsize=(20, 12), tight_layout=True, sharex=True, sharey=True)
        for s, sat_segments in enumerate(swarm_segments.values()):
            for d, segments in enumerate(sat_segments.values()):
                i = s * 2 * times.shape[0] + d * times.shape[0] + t
                plotting.plot_swarm_troughs_line(ax[s, d], segments[t]['mlat'], segments[t]['dne'],
                                                 segments[t]['smooth_dne'], swarm_troughs.iloc[i])
                ax[s, d].plot(mlat_profs[i], tec_profs[i], 'g--')
                ax[s, d].set_title(f"mlt: {swarm_troughs.iloc[i]['seg_e1_mlt']}, sat: {swarm_troughs.iloc[i]['sat']}")
                ax[s, d].grid(True)
        fig.savefig(f"E:\\temp_plots\\{t}_line.png")
        plt.close(fig)
