import numpy as np
import pandas
import traceback
from sklearn.metrics import confusion_matrix
import os
from scipy import stats

from ttools import io, swarm, rbf_inversion, utils, plotting
from ttools.config import mlat_grid, mlt_grid


def compare(times, tec_troughs, swarm_troughs, ssmlon, mlat_grid=mlat_grid, mlt_grid=mlt_grid):
    """
    info I want from each comparison:
        - yes/no for tec and swarm
        - pwall, ewall for tec and swarm
        - mlt / mlon of swarm pass
    """
    # get segment MLT
    t1 = swarm_troughs['seg_e1_mlt'].values * np.pi / 12
    t2 = swarm_troughs['seg_e2_mlt'].values * np.pi / 12
    seg_mlt = utils.average_angles(t1, t2) * 24 / np.pi

    # get mlat and trough slices
    print("Getting grid slices")
    data_grids = [mlat_grid, tec_troughs[swarm_troughs['tec_ind']]]
    mlat_profs, trough_profs = utils.get_grid_slice_line(swarm_troughs['seg_e1_mlt'], swarm_troughs['seg_e1_mlat'],
                                                         swarm_troughs['seg_e2_mlt'], swarm_troughs['seg_e2_mlat'],
                                                         data_grids, mlt_grid, mlat_grid)
    print("Comparing")
    results_list = []
    for t in range(swarm_troughs.shape[0]):
        tec_trough_mask = trough_profs[t] > 0
        tec_trough_found = tec_trough_mask.any()
        tec_e1 = 0
        tec_e2 = 0
        if tec_trough_found:
            tec_e1, tec_e2 = mlat_profs[t][np.argwhere(tec_trough_mask)[[0, -1], 0]]
        result = (tec_trough_found, tec_e1, tec_e2, times[swarm_troughs['tec_ind'][t]])
        results_list.append(result)

    # compile results
    pre_results = pandas.DataFrame(data=results_list, columns=['tec_trough', 'tec_e1', 'tec_e2', 'time'])
    results = pandas.DataFrame()
    down_mask = np.round(swarm_troughs['seg_e1_mlat'].values) == 75
    results['time'] = pre_results['time']
    results['tec_trough'] = pre_results['tec_trough']
    results['tec_ewall'] = pre_results['tec_e2'].where(down_mask, pre_results['tec_e1'])
    results['tec_pwall'] = pre_results['tec_e1'].where(down_mask, pre_results['tec_e2'])
    results['swarm_trough'] = swarm_troughs['trough']
    results['swarm_ewall'] = swarm_troughs['e2_mlat'].where(down_mask, swarm_troughs['e1_mlat'])
    results['swarm_pwall'] = swarm_troughs['e1_mlat'].where(down_mask, swarm_troughs['e2_mlat'])
    results['mlon'] = (15 * seg_mlt - 180 + ssmlon[swarm_troughs['tec_ind']] + 360) % 360
    return results


def run_single_day(date, bg_est_shape=(3, 15, 15), model_weight_max=20, rbf_bw=1, tv_hw=1, tv_vw=1, l2_weight=.1,
                   tv_weight=.05, perimeter_th=50, area_th=20, make_plots=False, plot_dir=None, artifact_key=None,
                   auroral_boundary=True, prior_order=1, prior='empirical', prior_arb_offset=-1):
    # setup time arrays
    one_h = np.timedelta64(1, 'h')
    start_time = date.astype('datetime64[D]').astype('datetime64[s]')
    end_time = start_time + np.timedelta64(1, 'D')
    comparison_times = np.arange(start_time, end_time, one_h)

    # get tec data, run trough detection algo
    tec_start = comparison_times[0] - np.floor(bg_est_shape[0] / 2) * one_h
    tec_end = comparison_times[-1] + (np.floor(bg_est_shape[0] / 2) + 1) * one_h
    tec, times, ssmlon, n = io.get_tec_data(tec_start, tec_end)
    ssmlon, = utils.moving_func_trim(bg_est_shape[0], ssmlon)
    artifacts = None
    arb = None
    if artifact_key is not None:
        artifacts = rbf_inversion.get_artifacts(mlt_grid, ssmlon, artifact_key)
    if auroral_boundary:
        arb, _ = io.get_arb_data(start_time, end_time)
    tec_troughs, x = rbf_inversion.get_tec_troughs(tec, times, bg_est_shape, model_weight_max, rbf_bw, tv_hw, tv_vw,
                                                   l2_weight, tv_weight, perimeter_th, area_th, artifacts, arb,
                                                   prior_order, prior, prior_arb_offset)

    print("Getting swarm troughs")
    swarm_segments = swarm.get_segments_data(comparison_times)
    swarm_troughs = swarm.get_swarm_troughs(swarm_segments)

    if make_plots:
        import matplotlib.pyplot as plt
        data_grids = [mlat_grid, x[swarm_troughs['tec_ind']]]
        mlat_profs, x_profs = utils.get_grid_slice_line(swarm_troughs['seg_e1_mlt'], swarm_troughs['seg_e1_mlat'],
                                                        swarm_troughs['seg_e2_mlt'], swarm_troughs['seg_e2_mlat'],
                                                        data_grids, mlt_grid, mlat_grid)
        trimmed_tec, = utils.moving_func_trim(bg_est_shape[0], tec)
        for t in range(len(comparison_times)):
            polar_fig, polar_ax = plt.subplots(1, 3, figsize=(20, 12), subplot_kw=dict(projection='polar'),
                                               tight_layout=True)
            line_fig, line_ax = plt.subplots(3, 2, figsize=(20, 12), tight_layout=True, sharex=True, sharey=True)
            swarm_troughs_t, swarm_segments_t, m_p, x_p = plotting.prepare_swarm_line_plot(t, swarm_segments,
                                                                                           swarm_troughs, mlat_profs,
                                                                                           x_profs)
            plotting.plot_all(polar_ax, line_ax, mlat_grid, mlt_grid, trimmed_tec[t], x[t],
                              swarm_troughs_t, tec_troughs[t], swarm_segments_t, m_p, x_p, ssmlon[t], arb[t])
            polar_fig.savefig(os.path.join(plot_dir, f"{comparison_times[t].astype('datetime64[D]')}_{t}_polar.png"))
            line_fig.savefig(os.path.join(plot_dir, f"{comparison_times[t].astype('datetime64[D]')}_{t}_line.png"))
            plt.close(polar_fig)
            plt.close(line_fig)

    return compare(comparison_times, tec_troughs, swarm_troughs, ssmlon)


def run_n_random_days(n, start_date=np.datetime64("2014-01-01"), end_date=np.datetime64("2020-01-01"),
                      bg_est_shape=(3, 15, 15), model_weight_max=20, rbf_bw=1, tv_hw=1, tv_vw=1, l2_weight=.1,
                      tv_weight=.05, perimeter_th=50, area_th=20, make_plots=False, plot_dir=None, artifact_key=None,
                      auroral_boundary=True, prior_order=1, prior='empirical', prior_arb_offset=-1):
    # get random days
    time_range_days = (end_date - start_date).astype('timedelta64[D]').astype(int)
    offsets = np.random.randint(0, time_range_days, n)
    dates = start_date + offsets.astype('timedelta64[D]')
    results = []
    for date in dates:
        print(f"Running {date}")
        try:
            result = run_single_day(date, bg_est_shape, model_weight_max, rbf_bw, tv_hw, tv_vw, l2_weight, tv_weight,
                                    perimeter_th, area_th, make_plots, plot_dir, artifact_key, auroral_boundary,
                                    prior_order, prior, prior_arb_offset)
            results.append(result)
        except Exception:
            traceback.print_exc()
    results = pandas.concat(results, ignore_index=True)
    return results


def process_results(results, good_mlon_range=None, bad_mlon_range=None):
    tn, fn, fp, tp = confusion_matrix(results['tec_trough'], results['swarm_trough']).ravel()
    acc = (tn + tp) / (tn + fn + fp + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fnr = 1 - tnr

    compare_mask = results['tec_trough'] & results['swarm_trough']
    pwall_diff = results['swarm_pwall'][compare_mask] - results['tec_pwall'][compare_mask]
    pwall_diff_mean = pwall_diff.mean()
    pwall_diff_std = pwall_diff.std()
    ewall_diff = results['swarm_ewall'][compare_mask] - results['tec_ewall'][compare_mask]
    ewall_diff_mean = ewall_diff.mean()
    ewall_diff_std = ewall_diff.std()

    statistics = {
        'tn': tn, 'fn': fn, 'fp': fp, 'tp': tp, 'acc': acc, 'tpr': tpr, 'tnr': tnr, 'fnr': fnr,
        'pwall_diff_mean': pwall_diff_mean, 'pwall_diff_std': pwall_diff_std, 'ewall_diff_mean': ewall_diff_mean,
        'ewall_diff_std': ewall_diff_std
    }

    if good_mlon_range is not None or bad_mlon_range is not None:
        if good_mlon_range is not None:
            mlon_mask = (results['mlon'] >= good_mlon_range[0]) & (results['mlon'] <= good_mlon_range[1])
        else:
            mlon_mask = ~((results['mlon'] >= bad_mlon_range[0]) & (results['mlon'] <= bad_mlon_range[1]))

        tn, fn, fp, tp = confusion_matrix(results['tec_trough'][mlon_mask], results['swarm_trough'][mlon_mask]).ravel()
        acc = (tn + tp) / (tn + fn + fp + tp)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fnr = 1 - tnr

        compare_mask = results['tec_trough'] & results['swarm_trough'] & mlon_mask
        pwall_diff = results['swarm_pwall'][compare_mask] - results['tec_pwall'][compare_mask]
        pwall_diff_mean = pwall_diff.mean()
        pwall_diff_std = pwall_diff.std()
        ewall_diff = results['swarm_ewall'][compare_mask] - results['tec_ewall'][compare_mask]
        ewall_diff_mean = ewall_diff.mean()
        ewall_diff_std = ewall_diff.std()

        mlon_restricted_statistics = {
            'mlon_tn': tn, 'mlon_fn': fn, 'mlon_fp': fp, 'mlon_tp': tp, 'mlon_acc': acc, 'mlon_tpr': tpr,
            'mlon_tnr': tnr, 'mlon_fnr': fnr, 'mlon_pwall_diff_mean': pwall_diff_mean,
            'mlon_pwall_diff_std': pwall_diff_std, 'mlon_ewall_diff_mean': ewall_diff_mean,
            'mlon_ewall_diff_std': ewall_diff_std
        }
        statistics.update(mlon_restricted_statistics)
    return statistics


def get_diffs(results, good_mlon_range=None, bad_mlon_range=None):
    compare_mask = results['tec_trough'] & results['swarm_trough']
    pwall_diff = results['swarm_pwall'][compare_mask] - results['tec_pwall'][compare_mask]
    ewall_diff = results['swarm_ewall'][compare_mask] - results['tec_ewall'][compare_mask]
    statistics = {'pwall_diff': pwall_diff, 'ewall_diff': ewall_diff}
    if good_mlon_range is not None or bad_mlon_range is not None:
        if good_mlon_range is not None:
            mlon_mask = (results['mlon'] >= good_mlon_range[0]) & (results['mlon'] <= good_mlon_range[1])
        else:
            mlon_mask = ~((results['mlon'] >= bad_mlon_range[0]) & (results['mlon'] <= bad_mlon_range[1]))
        statistics['mlon_mask'] = mlon_mask[compare_mask]

    return pandas.DataFrame(statistics)


PARAM_SAMPLING = {
    'tv_weight': stats.loguniform(.001, 1),
    'l2_weight': stats.loguniform(.001, 1),
    'bge_spatial_size': stats.randint(3, 13),
    'bge_temporal_rad': stats.randint(0, 3),
    'rbf_bw': stats.randint(1, 4),
    'tv_hw': stats.randint(1, 4),
    'tv_vw': stats.randint(1, 4),
    'model_weight_max': stats.randint(1, 20),
    'perimeter_th': stats.randint(10, 100),
    'area_th': stats.randint(10, 100),
    'artifact_key': [None, '3', '5', '7', '9'],
    'prior_order': [1, 2],
    'prior': ['empirical_model', 'auroral_boundary'],
    'prior_arb_offset': stats.randint(-5, 0),
}


def get_random_hyperparams():
    params = {}
    bge_temporal_size = 0
    bge_spatial_size = 0
    for p in PARAM_SAMPLING:
        if isinstance(PARAM_SAMPLING[p], list):
            val = np.random.choice(PARAM_SAMPLING[p], 1)[0]
        else:
            try:
                val = PARAM_SAMPLING[p].rvs().item()
            except:
                val = PARAM_SAMPLING[p].rvs()
        if p == 'bge_temporal_rad':
            bge_temporal_size = val * 2 + 1
        elif p == 'bge_spatial_size':
            bge_spatial_size = val * 2 + 1
        else:
            params[p] = val
    params['bg_est_shape'] = (bge_temporal_size, bge_spatial_size, bge_spatial_size)
    return params