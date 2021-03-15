import numpy as np
import pandas
import pytest
import tempfile
import os

from ttools import compare

T = 7
nmlt = 40
nmlat = 20


@pytest.fixture(scope='module')
def trough_setup():
    """perfect match, swarm bigger, tec bigger, offset, swarm, tec, neither"""
    mlt_grid, mlat_grid = np.meshgrid(np.arange(-12, 12, 24 / nmlt), np.arange(30, 90, 60 / nmlat))
    swarm_troughs = pandas.DataFrame(index=np.arange(T))
    swarm_troughs['sat'] = 'A'
    swarm_troughs['seg_e1_mlat'] = 45
    swarm_troughs['seg_e2_mlat'] = 75
    swarm_troughs['seg_e1_mlt'] = 0
    swarm_troughs['seg_e2_mlt'] = 0
    swarm_troughs['tec_ind'] = np.arange(T)
    min_dne = np.zeros(7)
    min_dne[:5] = -.5
    swarm_troughs['min_dne'] = min_dne
    swarm_troughs['direction'] = 'up'

    trough = np.zeros(T, dtype=bool)
    trough[:5] = True
    #                   match             swarm big         tec big           offset         swarm tec neither
    e1_mlat = np.array([mlat_grid[10, 0], mlat_grid[9, 0], mlat_grid[10, 0], mlat_grid[10, 0], 50., 0., 0.])
    e2_mlat = np.array([mlat_grid[11, 0], mlat_grid[12, 0], mlat_grid[11, 0], mlat_grid[11, 0], 60., 0., 0.])
    e1_mlt = np.zeros(T)
    e2_mlt = np.zeros(T)

    swarm_troughs['trough'] = trough
    swarm_troughs['e1_mlat'] = e1_mlat
    swarm_troughs['e1_mlt'] = e1_mlt
    swarm_troughs['e2_mlat'] = e2_mlat
    swarm_troughs['e2_mlt'] = e2_mlt

    tec_troughs = np.zeros((T, nmlat, nmlt), dtype=bool)
    tec_troughs[0, 10:12, 10:30] = True  # match
    tec_troughs[1, 10:12, 10:30] = True  # swarm big
    tec_troughs[2, 9:13, 10:30] = True  # tec big
    tec_troughs[3, 10:12, 30:] = True  # offset
    # swarm
    tec_troughs[5, 10:12, 10:30] = True  # tec
    # neither
    yield tec_troughs, swarm_troughs, mlat_grid, mlt_grid


@pytest.fixture(scope='module')
def results(trough_setup):
    tec_troughs, swarm_troughs, mlat_grid, mlt_grid = trough_setup
    ssmlon = (np.arange(tec_troughs.shape[0]) * 90) % 360
    times = np.datetime64("2000-01-01T00:00:00") + np.arange(tec_troughs.shape[0]) * np.timedelta64(1, 'h')
    results = compare.compare(times, tec_troughs, swarm_troughs, ssmlon, mlat_grid, mlt_grid)
    yield results


@pytest.fixture(scope='module')
def dmlat(trough_setup):
    tec_troughs, swarm_troughs, mlat_grid, mlt_grid = trough_setup
    yield mlat_grid[1, 0] - mlat_grid[0, 0]


def test_compare(results):
    assert np.all(results['swarm_trough'] == [True, True, True, True, True, False, False])
    assert np.all(results['tec_trough'] == [True, True, True, False, False, True, False])
    assert np.all(results['mlon'] == (180 + np.arange(T) * 90) % 360)
    assert results['tec_ewall'][0] == results['swarm_ewall'][0]
    assert results['tec_pwall'][0] == results['swarm_pwall'][0]
    assert results['tec_ewall'][1] > results['swarm_ewall'][1]
    assert results['tec_pwall'][1] < results['swarm_pwall'][1]
    assert results['tec_ewall'][2] < results['swarm_ewall'][2]
    assert results['tec_pwall'][2] > results['swarm_pwall'][2]


def test_get_diffs(results, dmlat):
    diffs = compare.get_diffs(results)
    assert np.all(diffs['pwall_diff'] == [0, dmlat, -dmlat])
    assert np.all(diffs['ewall_diff'] == [0, -dmlat, dmlat])


def test_get_comparison_stats(results, dmlat):
    statistics = compare.get_comparison_stats(results)
    assert statistics['tn'] == 1
    assert statistics['fn'] == 2
    assert statistics['fp'] == 1
    assert statistics['tp'] == 3
    assert statistics['acc'] == 4 / 7
    assert statistics['pwall_diff_mean'] == 0
    assert statistics['ewall_diff_mean'] == 0
    assert statistics['pwall_diff_std'] == np.std([0, 3, -3], ddof=1)
    assert statistics['ewall_diff_std'] == np.std([0, -3, 3], ddof=1)

    r = [175, 185]

    keep_mlon_mask = (results['mlon'] >= r[0]) & (results['mlon'] <= r[1])
    statistics = compare.get_comparison_stats(results[keep_mlon_mask])
    assert statistics['tn'] == 0
    assert statistics['fn'] == 1
    assert statistics['fp'] == 0
    assert statistics['tp'] == 1
    assert statistics['acc'] == 1 / 2
    assert statistics['pwall_diff_mean'] == 0
    assert statistics['ewall_diff_mean'] == 0
    assert np.isnan(statistics['pwall_diff_std'])
    assert np.isnan(statistics['ewall_diff_std'])

    reject_mlon_mask = ~((results['mlon'] >= r[0]) & (results['mlon'] <= r[1]))
    statistics = compare.get_comparison_stats(results[reject_mlon_mask])
    assert statistics['tn'] == 1
    assert statistics['fn'] == 1
    assert statistics['fp'] == 1
    assert statistics['tp'] == 2
    assert statistics['acc'] == 3 / 5
    assert statistics['pwall_diff_mean'] == 0
    assert statistics['ewall_diff_mean'] == 0
    assert statistics['pwall_diff_std'] == np.std([3, -3], ddof=1)
    assert statistics['ewall_diff_std'] == np.std([-3, 3], ddof=1)


def test_process_results(results):
    statistics = compare.process_results(results, good_mlon_range=[175, 185])
    assert statistics['tn'] == 1
    assert statistics['fn'] == 2
    assert statistics['fp'] == 1
    assert statistics['tp'] == 3
    assert statistics['acc'] == 4 / 7
    assert statistics['pwall_diff_mean'] == 0
    assert statistics['ewall_diff_mean'] == 0
    assert statistics['pwall_diff_std'] == np.std([0, 3, -3], ddof=1)
    assert statistics['ewall_diff_std'] == np.std([0, -3, 3], ddof=1)
    assert statistics['mlon_tn'] == 0
    assert statistics['mlon_fn'] == 1
    assert statistics['mlon_fp'] == 0
    assert statistics['mlon_tp'] == 1
    assert statistics['mlon_acc'] == 1 / 2
    assert statistics['mlon_pwall_diff_mean'] == 0
    assert statistics['mlon_ewall_diff_mean'] == 0
    assert np.isnan(statistics['mlon_pwall_diff_std'])
    assert np.isnan(statistics['mlon_ewall_diff_std'])

    statistics = compare.process_results(results, bad_mlon_range=[175, 185])
    assert statistics['mlon_tn'] == 1
    assert statistics['mlon_fn'] == 1
    assert statistics['mlon_fp'] == 1
    assert statistics['mlon_tp'] == 2
    assert statistics['mlon_acc'] == 3 / 5
    assert statistics['mlon_pwall_diff_mean'] == 0
    assert statistics['mlon_ewall_diff_mean'] == 0
    assert statistics['mlon_pwall_diff_std'] == np.std([3, -3], ddof=1)
    assert statistics['mlon_ewall_diff_std'] == np.std([-3, 3], ddof=1)


def test_run_single_day():
    date = np.datetime64("2015-10-07")
    low_reg_stats = compare.get_comparison_stats(compare.run_single_day(date, model_weight_max=5, l2_weight=.05,
                                                                        tv_weight=.01))
    with tempfile.TemporaryDirectory() as tempdir:
        high_reg_stats = compare.get_comparison_stats(compare.run_single_day(date, model_weight_max=10, l2_weight=.5,
                                                                             tv_weight=.1, plot_dir=tempdir,
                                                                             make_plots=True))
        assert len(os.listdir(tempdir)) == 48
    assert low_reg_stats['tpr'] >= high_reg_stats['tpr']
    assert low_reg_stats['fpr'] >= high_reg_stats['fpr']
    assert low_reg_stats['fnr'] <= high_reg_stats['fnr']
    assert low_reg_stats['tnr'] <= high_reg_stats['tnr']


def test_random_parameter_search():
    start_date = np.datetime64("2015-10-07")
    end_date = np.datetime64("2015-10-08")
    with tempfile.TemporaryDirectory() as tempdir:
        compare.random_parameter_search(1, 1, tempdir, start_date, end_date)
        overall_stats_fn = os.path.join(tempdir, 'results.csv')
        exp_0_results_fn = os.path.join(tempdir, 'experiment_0', 'results.csv')
        exp_0_params_fn = os.path.join(tempdir, 'experiment_0', 'params.yaml')
        for f in [overall_stats_fn, exp_0_params_fn, exp_0_results_fn]:
            assert os.path.exists(f)
        overall_stats = pandas.read_csv(overall_stats_fn)
        assert overall_stats.shape == (1, 27)
        exp_0_results = pandas.read_csv(exp_0_results_fn)
        assert exp_0_results.shape == (24 * 6, 13)
