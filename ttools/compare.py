"""
parameter search:
    Process:
        - choose days
        - run swarm for all those days
        - for n random parameter settings:
            - run algo on same days
            - evaluate performance and append to results list
"""
import numpy as np
import pandas
from sklearn.metrics import confusion_matrix
import os
from scipy import stats
import time
import itertools
from matplotlib import pyplot as plt

from ttools import io, satellite, utils, config, convert, tec as ttec, plotting
from ttools.trough_labeling import TroughLabelJobManager


def compare(times, tec_troughs, swarm_troughs, ssmlon, swarm_grid_x=None, swarm_grid_y=None, mlat_grid=None, mlt_grid=None, mlat_profs=None):
    _, unique_swarm_idx, inv = np.unique(swarm_troughs['sat_ind'], return_index=True, return_inverse=True)
    if mlat_grid is None:
        mlat_grid = config.mlat_grid
    if mlt_grid is None:
        mlt_grid = config.mlt_grid
    if swarm_grid_x is None or swarm_grid_y is None:
        swarm_grid_x, swarm_grid_y = get_swarm_grid_cells(swarm_troughs.iloc[unique_swarm_idx], mlat_grid, mlt_grid)
    if mlat_profs is None:
        mlat_profs = get_profs(mlat_grid, swarm_grid_x, swarm_grid_y)

    trough_profs = get_profs(tec_troughs[swarm_troughs['tec_ind'].values[unique_swarm_idx]], swarm_grid_x, swarm_grid_y)
    print("Comparing")
    tec_trough_found = np.zeros(len(trough_profs), dtype=bool)
    tec_e1 = np.zeros(len(trough_profs))
    tec_e2 = np.zeros(len(trough_profs))
    for t in range(len(trough_profs)):
        tec_trough_mask = trough_profs[t] > 0
        tec_trough_found[t] = tec_trough_mask.any()
        if tec_trough_found[t]:
            tec_e1[t], tec_e2[t] = mlat_profs[t][np.argwhere(tec_trough_mask)[[0, -1], 0]]
    return compile_comparison_results(times, swarm_troughs, tec_trough_found[inv], tec_e1[inv], tec_e2[inv], ssmlon)


def get_profs(data_grid, grid_cell_x, grid_cell_y, reduce_func=np.nanmean):
    # new grid for each grid cell list
    profs = []
    if data_grid.ndim == 3:
        assert data_grid.shape[0] == len(grid_cell_x)
        for t in range(len(grid_cell_x)):
            profs.append(reduce_func(data_grid[t][grid_cell_y[t], grid_cell_x[t]], axis=1))
    else:
        for t in range(len(grid_cell_x)):
            profs.append(reduce_func(data_grid[grid_cell_y[t], grid_cell_x[t]], axis=1))
    return profs


def get_swarm_grid_cells(swarm_troughs, mlat_grid=None, mlt_grid=None, linewidth=3):
    if mlat_grid is None:
        mlat_grid = config.mlat_grid
    if mlt_grid is None:
        mlt_grid = config.mlt_grid
    # get mlat and trough slices
    print("Getting grid slices")
    idx_grids = np.meshgrid(np.arange(mlt_grid.shape[1]), np.arange(mlt_grid.shape[0]))
    e1_mlat = swarm_troughs['seg_e1_mlat'].values * 11 / 6 + 30 - 45 * 11 / 6
    e2_mlat = swarm_troughs['seg_e2_mlat'].values * 11 / 6 + 30 - 45 * 11 / 6
    swarm_grid_x, swarm_grid_y = utils.get_grid_slice_line(swarm_troughs['seg_e1_mlt'].values, e1_mlat,
                                                           swarm_troughs['seg_e2_mlt'].values, e2_mlat, idx_grids,
                                                           mlt_grid, mlat_grid, reduce_func=None, order=0,
                                                           linewidth=linewidth)
    return [s.astype(int) for s in swarm_grid_x], [s.astype(int) for s in swarm_grid_y]


def compile_comparison_results(times, swarm_troughs, tec_trough_found, tec_e1, tec_e2, ssmlon):
    # get segment MLT
    t1 = swarm_troughs['seg_e1_mlt'].values * np.pi / 12
    t2 = swarm_troughs['seg_e2_mlt'].values * np.pi / 12
    seg_mlt = utils.average_angles(t1, t2) * 12 / np.pi
    # compile results
    pre_results = pandas.DataFrame()
    pre_results['time'] = times[swarm_troughs['tec_ind']]
    pre_results['sat'] = swarm_troughs['sat']
    pre_results['swarm_dne'] = swarm_troughs['min_dne']
    pre_results['direction'] = swarm_troughs['direction']
    pre_results['tec_trough'] = tec_trough_found
    pre_results['tec_ewall'] = np.where(swarm_troughs['direction'] == 'up', tec_e1, tec_e2)
    pre_results['tec_pwall'] = np.where(swarm_troughs['direction'] == 'up', tec_e2, tec_e1)
    pre_results['swarm_trough'] = swarm_troughs['trough']
    pre_results['swarm_ewall'] = np.where(swarm_troughs['direction'] == 'up', swarm_troughs['e1_mlat'],
                                          swarm_troughs['e2_mlat'])
    pre_results['swarm_pwall'] = np.where(swarm_troughs['direction'] == 'up', swarm_troughs['e2_mlat'],
                                          swarm_troughs['e1_mlat'])
    pre_results['mlt'] = seg_mlt
    if ssmlon is not None:
        pre_results['mlon'] = convert.mlt_to_mlon_sub(seg_mlt, ssmlon[swarm_troughs['tec_ind']])
    pre_results['id'] = np.arange(swarm_troughs.shape[0])

    # process the pre_results: output should be a single up / down for each satellite for each tec map
    results = []
    for i in np.unique(swarm_troughs['sat_ind'].values):
        mask = swarm_troughs['sat_ind'] == i
        candidates = pre_results[mask]
        if not candidates['tec_trough'].iloc[0]:
            # if no tec trough, use normal swarm defn
            if candidates['swarm_trough'].any():
                results.append(candidates[candidates['swarm_trough']].sort_values('swarm_ewall').iloc[0])
            else:
                results.append(candidates.iloc[0])
        else:
            # if there is tec trough, find best swarm trough
            if not candidates['swarm_trough'].any():
                results.append(candidates.iloc[0])
            else:
                scores = abs(candidates['swarm_ewall'] - candidates['tec_ewall']) + abs(candidates['swarm_pwall'] - candidates['tec_pwall'])
                idx = np.argmin(scores.values[candidates['swarm_trough'].values])
                results.append(candidates[candidates['swarm_trough']].iloc[idx])
    return pandas.DataFrame(results)


def get_comparison_stats(results):
    tn, fn, fp, tp = confusion_matrix(results['tec_trough'], results['swarm_trough']).ravel()
    acc = (tn + tp) / (tn + fn + fp + tp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    diffs = get_diffs(results)
    pwall_diff_mean = diffs['pwall_diff'].mean()
    pwall_diff_std = diffs['pwall_diff'].std()
    ewall_diff_mean = diffs['ewall_diff'].mean()
    ewall_diff_std = diffs['ewall_diff'].std()

    statistics = {
        'tn': tn, 'fn': fn, 'fp': fp, 'tp': tp, 'acc': acc, 'tpr': tpr, 'tnr': tnr, 'fnr': fnr, 'fpr': fpr,
        'pwall_diff_mean': pwall_diff_mean, 'pwall_diff_std': pwall_diff_std, 'ewall_diff_mean': ewall_diff_mean,
        'ewall_diff_std': ewall_diff_std
    }
    return statistics


def process_results(results, good_mlon_range=None, bad_mlon_range=None):
    statistics = get_comparison_stats(results)

    if good_mlon_range is not None or bad_mlon_range is not None:
        if good_mlon_range is not None:
            mlon_mask = (results['mlon'] >= good_mlon_range[0]) & (results['mlon'] <= good_mlon_range[1])
        else:
            mlon_mask = ~((results['mlon'] >= bad_mlon_range[0]) & (results['mlon'] <= bad_mlon_range[1]))
        mlon_restricted_stats = {f'mlon_{k}': v for k, v in get_comparison_stats(results[mlon_mask]).items()}
        statistics.update(mlon_restricted_stats)
    return statistics


def get_diffs(results):
    compare_mask = results['tec_trough'] & results['swarm_trough']
    pwall_diff = results['tec_pwall'][compare_mask] - results['swarm_pwall'][compare_mask]
    ewall_diff = results['tec_ewall'][compare_mask] - results['swarm_ewall'][compare_mask]
    statistics = {'pwall_diff': pwall_diff, 'ewall_diff': ewall_diff}
    return pandas.DataFrame(statistics)


def random_parameter_search(job_class, n_experiments, n_trials, base_dir="E:\\trough_comparison"):
    results_file = os.path.join(base_dir, "results.csv")
    if os.path.isfile(results_file):
        processed_results = pandas.read_csv(results_file, index_col=0).to_dict('records')
    else:
        processed_results = []
    prev_max_exp = max([int(d[11:]) for d in os.listdir(base_dir) if not os.path.splitext(d)[1]])
    # get random days
    comparison_manager = ComparisonManager.random_dates(n_trials)
    for i in range(n_experiments):
        # setup directory
        experiment_dir = os.path.join(base_dir, f"experiment_{i + prev_max_exp + 1}")
        os.makedirs(experiment_dir, exist_ok=True)
        job_manager = TroughLabelJobManager.get_random(job_class, experiment_dir)
        t0 = time.time()
        results = comparison_manager.run_comparison(job_manager)
        statistics = process_results(results, bad_mlon_range=[130, 260])
        for k, v in statistics.items():
            print(k, v)
        results.to_csv(os.path.join(experiment_dir, "results.csv"))
        tf = time.time()
        print(f"THAT TOOK {(tf - t0) / 60} MINUTES")
        processed_results.append(statistics)
        pandas.DataFrame(processed_results).to_csv(os.path.join(base_dir, "results.csv"))


def grid_parameter_search(default_params, parameter_values, n_trials, base_dir="E:\\trough_comparison",
                          start_date=np.datetime64("2014-01-01"), end_date=np.datetime64("2020-01-01")):
    processed_results = []
    grid_values = list(itertools.product(*parameter_values.values()))
    grid_keys = [list(parameter_values.keys())] * len(grid_values)
    i = 0
    for grid_key, grid_value in zip(grid_keys, grid_values):
        # setup directory
        experiment_dir = os.path.join(base_dir, f"experiment_{i}")
        os.makedirs(experiment_dir, exist_ok=True)

        # get and save hyperparameter list
        params = default_params.copy()
        update = {key: value for key, value in zip(grid_key, grid_value)}
        params.update(update)
        print(params)
        io.write_yaml(os.path.join(experiment_dir, 'params.yaml'), **params)

        t0 = time.time()
        results = run_n_random_days(n_trials, **params, start_date=start_date, end_date=end_date)
        statistics = process_results(results, bad_mlon_range=[130, 260])
        for k, v in statistics.items():
            print(k, v)
        results.to_csv(os.path.join(experiment_dir, "results.csv"))
        tf = time.time()
        print(f"THAT TOOK {(tf - t0) / 60} MINUTES")
        processed_results.append(statistics)
        pandas.DataFrame(processed_results).to_csv(os.path.join(base_dir, "results.csv"))

        i += 1


class ComparisonManager:

    def __init__(self, dates, plot_dir=None):
        self.dates = dates
        # get swarm trough candidates
        print("Getting SWARM trough candidates...")
        swarm_troughs = []
        times = []
        for date in dates:
            t = np.arange(date, date + np.timedelta64(1, 'D'), np.timedelta64(1, 'h'))
            swarm_segments = satellite.get_segments_data(t, 'swarm')
            swarm_trough = satellite.get_troughs(swarm_segments)
            swarm_troughs.append(swarm_trough)
            times.append(t)
            if plot_dir is not None:
                for i in range(24):
                    troughs = swarm_trough[swarm_trough['tec_ind'] == i]
                    fig, ax = plt.subplots(3, 2, figsize=(16, 10), tight_layout=True, sharex=True, sharey=True)
                    for s, (sat, sat_segments) in enumerate(swarm_segments.items()):
                        sat_troughs = troughs[troughs['sat'] == sat]
                        for d, (direction, segments) in enumerate(sat_segments.items()):
                            if i >= len(segments):
                                break
                            tcs = sat_troughs[sat_troughs['direction'] == direction]
                            ax[s, d].plot(segments[i]['mlat'], segments[i]['smooth_dne'])
                            ax[s, d].plot(segments[i]['mlat'], segments[i]['dne'], 'k.', ms=1)
                            for _, tc in tcs[1:].iterrows():
                                ax[s, d].plot(tc[['e1_mlat', 'e2_mlat']], [0, 0], 'r', linestyle='solid', marker='|', ms=10)
                                ax[s, d].plot(tc['min_mlat'], tc['min_dne'], 'rx')
                            ax[s, d].set_title(f"mlt: {tcs['seg_e1_mlt'].iloc[0]}, sat: {sat}")
                            ax[s, d].grid(True)
                    fig.savefig(os.path.join(plot_dir, f"{date.astype('datetime64[D]')}_{i}_swarm.png"))
                    plt.close(fig)
        self.swarm_troughs = pandas.concat(swarm_troughs, ignore_index=True)
        self.swarm_troughs = satellite.fix_trough_list(self.swarm_troughs)
        _, self.unique_swarm_idx = np.unique(self.swarm_troughs['sat_ind'].values, return_index=True)
        self.times = np.concatenate(times, axis=0)
        self.swarm_grid_x, self.swarm_grid_y = get_swarm_grid_cells(self.swarm_troughs.iloc[self.unique_swarm_idx],
                                                                    config.mlat_grid, config.mlt_grid)
        self.ssmlon = None
        self.plot_dir = plot_dir

    @classmethod
    def random_dates(cls, n, **kwargs):
        dates = utils.get_random_dates(n)
        return cls(dates, **kwargs)

    def _get_tec_troughs_no_opt(self, job_manager):
        troughs = []
        ssmlon = []
        for d, date in enumerate(self.dates):
            job = job_manager.make_job(date)
            job.run()
            if self.plot_dir is not None:
                mask = (self.swarm_troughs['tec_ind'] < (d + 1) * 24) & (self.swarm_troughs['tec_ind'] >= d * 24)
                job.plot(self.swarm_troughs[mask], self.plot_dir)
            troughs.append(job.trough)
            if self.ssmlon is None:
                ssmlon.append(job.ssmlon)
        if self.ssmlon is None:
            self.ssmlon = np.concatenate(ssmlon, axis=0)
        return np.concatenate(troughs, axis=0)

    def _get_tec_troughs_opt(self, job_manager):
        data = {'model_output': [], 'arb': []}
        ssmlon = []
        # get pre threshold output
        for date in self.dates:
            job = job_manager.make_job(date)
            job.run()
            for d in data:
                data[d].append(getattr(job, d))
            if self.ssmlon is None:
                ssmlon.append(job.ssmlon)
        if self.ssmlon is None:
            self.ssmlon = np.concatenate(ssmlon, axis=0)
        print("Concatenating...")
        model_output, arb = utils.concatenate(data['model_output'], data['arb'])
        print("Optimizing threshold...")
        alpha_1 = .15
        alpha_2 = .15
        thresholds = np.linspace(0, 1.5, 20)
        loss = np.zeros_like(thresholds)
        for i, t in enumerate(thresholds):
            tec_troughs = model_output >= t
            tec_troughs = ttec.postprocess(tec_troughs, job_manager.perimeter_th, job_manager.area_th, arb,
                                           job_manager.closing_rad)
            results = self._compare(tec_troughs)
            mask = results['tec_trough'] & results['swarm_trough']
            acc = np.mean(results['tec_trough'] == results['swarm_trough'])
            ploss = alpha_1 * np.nanmean(abs(results[mask]['tec_pwall'] - results[mask]['swarm_pwall']))
            eloss = alpha_2 * np.nanmean(abs(results[mask]['tec_ewall'] - results[mask]['swarm_ewall']))
            closs = 1 - acc
            loss[i] = ploss + eloss + closs
            print(t, loss[i], ploss, eloss, closs, mask.sum())
        threshold = thresholds[np.nanargmin(loss)]
        tec_troughs = model_output >= threshold
        tec_troughs = ttec.postprocess(tec_troughs, job_manager.perimeter_th, job_manager.area_th, arb)
        return tec_troughs

    def _compare(self, tec_troughs):
        return compare(self.times, tec_troughs, self.swarm_troughs, self.ssmlon, self.swarm_grid_x, self.swarm_grid_y)

    def get_tec_troughs(self, job_manager):
        if job_manager.threshold is None:
            return self._get_tec_troughs_opt(job_manager)
        else:
            return self._get_tec_troughs_no_opt(job_manager)

    def run_comparison(self, job_manager):
        tec_troughs = self.get_tec_troughs(job_manager)
        return self._compare(tec_troughs)
