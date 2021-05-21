import numpy as np
import pandas
import os
import matplotlib.pyplot as plt

from ttools import io, utils, plotting, config, swarm, compare, tec as ttec
from ttools.trough_labeling import rbf_inversion


def run_single(date):
    # get tec data, run trough detection algo
    bg_est_shape = (1, 17, 11)
    one_h = np.timedelta64(1, 'h')
    tec_start = date - np.floor(bg_est_shape[0] / 2) * one_h
    tec_end = date + np.timedelta64(1, 'D')
    tec, times, ssmlon, n = io.get_tec_data(tec_start, tec_end)
    arb, _ = io.get_arb_data(tec_start, tec_end)

    x, times = ttec.preprocess_interval(tec, times, bg_est_shape=bg_est_shape)
    # setup optimization
    print("Setting up inversion optimization")
    args = rbf_inversion.get_optimization_args(x, times, model_weight_max=15, l2_weight=.05, tv_weight=.15,
                                               prior_order=1, prior='auroral_boundary', arb=arb, arb_offset=-3)
    # run optimization
    print("Running inversion optimization")
    model_output = rbf_inversion.run_multiple(args, parallel=config.PARALLEL)

    swarm_segments = swarm.get_segments_data(times)
    swarm_troughs = swarm.get_swarm_troughs(swarm_segments)

    return model_output, swarm_segments, swarm_troughs, ssmlon


if __name__ == "__main__":
    start_date = np.datetime64("2014-01-01")
    end_date = np.datetime64("2020-01-01")
    time_range_days = (end_date - start_date).astype('timedelta64[D]').astype(int)
    offsets = np.random.randint(0, time_range_days, 50)
    dates = start_date + offsets.astype('timedelta64[D]')

    thresholds = np.arange(-1, 5, .1)

    model_outputs = []
    swarm_troughs = []
    times = []
    ssmlons = []
    swarm_times = []
    for date in dates:
        model, swarm_segments, swarm_trough, ssmlon = run_single(date)
        time = np.arange(date, date + np.timedelta64(1, 'D'), np.timedelta64(1, 'h'))
        model_outputs.append(model)
        swarm_troughs.append(swarm_trough)
        times.append(time)
        ssmlons.append(ssmlon)
        swarm_times.append(time[swarm_trough['tec_ind'].values])
    swarm_troughs = pandas.concat(swarm_troughs, ignore_index=True)
    model_outputs = np.concatenate(model_outputs, axis=0)
    times = np.concatenate(times)
    ssmlons = np.concatenate(ssmlons)
    swarm_times = np.concatenate(swarm_times)

    swarm_troughs['tec_ind'] = swarm_troughs['tec_ind'].values + 24 * (np.argmax(swarm_times[:, None] == times[None, :], axis=1) // 24)

    swarm_grid_x, swarm_grid_y = compare.get_swarm_grid_cells(swarm_troughs, config.mlat_grid, config.mlt_grid)
    model_profs = compare.get_profs(model_outputs[swarm_troughs['tec_ind'].values], swarm_grid_x, swarm_grid_y)
    mlat_profs = compare.get_profs(config.mlat_grid, swarm_grid_x, swarm_grid_y)
    theta = 0
    alpha_1 = 1
    alpha_2 = 1
    step_size = .001
    hist = {'theta': [], 'loss': [], 'tp': []}
    for i in range(25):
        tec_troughs = model_outputs >= theta
        results = compare.compare(times, tec_troughs, swarm_troughs, ssmlons, swarm_grid_x, swarm_grid_y)
        mask = results['tec_trough'] & results['swarm_trough']
        pwall_grad = np.array([1 / np.minimum(-.01, np.gradient(model_profs[row['id']])[np.argmax(mlat_profs[row['id']] == row['tec_pwall'])]) for _, row in results[mask].iterrows()])
        ewall_grad = np.array([1 / np.maximum(.01, np.gradient(model_profs[row['id']])[np.argmax(mlat_profs[row['id']] == row['tec_ewall'])]) for _, row in results[mask].iterrows()])

        loss = np.mean(alpha_1 * (results[mask]['tec_pwall'] - results[mask]['swarm_pwall'])**2 + alpha_2 * (results[mask]['tec_ewall'] - results[mask]['swarm_ewall'])**2)
        pwall_loss_grad = alpha_1 * (results[mask]['tec_pwall'] - results[mask]['swarm_pwall']) * pwall_grad
        ewall_loss_grad = alpha_2 * (results[mask]['tec_ewall'] - results[mask]['swarm_ewall']) * ewall_grad
        dtheta = np.mean(pwall_loss_grad.values + ewall_loss_grad.values)
        theta -= step_size * dtheta
        print(theta, loss, mask.sum())
        hist['theta'].append(theta)
        hist['loss'].append(loss)
        hist['tp'].append(mask.sum())

    plt.style.use('ggplot')
    fig, ax = plt.subplots(3, 1, sharex=True, tight_layout=True)
    ax[0].plot(hist['theta'], label="theta")
    ax[1].plot(hist['loss'], label="loss")
    ax[2].plot(hist['tp'], label="True Positives")
    plt.legend()
    plt.show()
