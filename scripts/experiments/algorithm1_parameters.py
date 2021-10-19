import numpy as np
import os
import matplotlib.pyplot as plt

from ttools import io, utils, plotting, config
from ttools.trough_labeling import rbf_inversion


def run_single(date, i, bg_est_shape, model_weight_max, rbf_bw, tv_hw, tv_vw, l2_weight, tv_weight, perimeter_th,
               area_th, prior_order, prior, prior_arb_offset):
    job = rbf_inversion.RbfInversionLabelJob(date, bg_est_shape, model_weight_max, rbf_bw, tv_hw, tv_vw, l2_weight,
                                             tv_weight, prior_order, prior, prior_arb_offset, perimeter_th, area_th,
                                             threshold=1)
    job.run()

    polar_fig, polar_ax = plt.subplots(1, 3, figsize=(18, 10), subplot_kw=dict(projection='polar'), tight_layout=True)
    plotting.polar_pcolormesh(polar_ax[0], config.mlat_grid, config.mlt_grid, job.x[0], vmin=-.5, vmax=.5, cmap='coolwarm')
    plotting.polar_pcolormesh(polar_ax[1], config.mlat_grid, config.mlt_grid, job.model_output[0], vmin=-2, vmax=2, cmap='coolwarm')
    plotting.polar_pcolormesh(polar_ax[2], config.mlat_grid, config.mlt_grid, job.trough[0], vmin=0, vmax=1, cmap='Blues')
    plotting.plot_arb(polar_ax[0], config.mlt_grid[0], job.arb[0])
    plotting.plot_arb(polar_ax[1], config.mlt_grid[0], job.arb[0])
    plotting.format_polar_mag_ax(polar_ax)
    polar_fig.savefig(os.path.join("C:\\Users\\Greg\\Desktop\\study plots\\a1_params", f"{i}.png"))
    plt.close(polar_fig)


if __name__ == "__main__":
    config.PARALLEL = False

    default_params = {
        'bg_est_shape': (1, 17, 17),
        'model_weight_max': 10,
        'rbf_bw': 1,
        'tv_hw': 2,
        'tv_vw': 1,
        'l2_weight': .05,
        'tv_weight': .1,
        'perimeter_th': 50,
        'area_th': 50,
        'prior_order': 1,
        'prior': 'auroral_boundary',
        'prior_arb_offset': -3
    }

    non_default_params = {
        'bg_est_shape': [(1, 13, 13), (1, 15, 15), (1, 19, 19), (1, 21, 21), (3, 17, 17), (3, 13, 13), (3, 15, 15), (3, 19, 19), (3, 21, 21)],
        'model_weight_max': [2, 5, 15, 20, 25],
        'rbf_bw': [1, 2],
        'tv_hw': [1, 2, 3, 4, 5],
        'tv_vw': [1, 2, 3, 4, 5],
        'l2_weight': [.01, .1, .2],
        'tv_weight': [.01, .05, .2],
        'area_th': [10, 100],
        'prior_order': [1, 2],
        'prior': ['auroral_boundary', 'empirical_model'],
        'prior_arb_offset': [-2, -5, -7]
    }

    one_h = np.timedelta64(1, 'h')
    date = np.datetime64("2014-02-19T03:00:00")

    i = 0
    run_single(date, i, **default_params)

    param_output = []
    for param_name, param_value in non_default_params.items():
        for val in param_value:
            i += 1
            param_output.append((i, param_name, val))
            params = default_params.copy()
            params.update({param_name: val})
            run_single(date, i, **params)

    tec, times, ssmlon, n = io.get_tec_data(date, date + one_h)
    arb, _ = io.get_arb_data(date, date + one_h)

    polar_fig, polar_ax = plt.subplots(1, 1, figsize=(18, 10), subplot_kw=dict(projection='polar'), tight_layout=True)
    pcm = plotting.polar_pcolormesh(polar_ax, config.mlat_grid, config.mlt_grid, tec[0], vmin=0, vmax=25)
    plotting.plot_arb(polar_ax, config.mlt_grid[0], arb[0])
    plt.colorbar(pcm)
    plotting.format_polar_mag_ax(polar_ax)
    polar_fig.savefig(os.path.join("C:\\Users\\Greg\\Desktop\\study plots\\a1_params", f"base.png"))
    plt.close(polar_fig)

    for p in param_output:
        print(*p)
