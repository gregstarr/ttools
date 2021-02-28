import numpy as np
import pandas
import pytest

from ttools import compare


@pytest.fixture()
def trough_setup():
    """perfect match, swarm bigger, tec bigger, offset, swarm, tec, neither"""
    T = 7
    nmlt = 40
    nmlat = 20
    mlt_grid, mlat_grid = np.meshgrid(np.arange(-12, 12, 24 / nmlt), np.arange(30, 90, 60 / nmlat))
    columns = ['seg_e1_mlat', 'seg_e1_mlt', 'seg_e2_mlat', 'seg_e2_mlt', 'trough', 'e1_mlat', 'e2_mlat', 'e1_mlt',
               'e2_mlt', 'sat', 'tec_ind']
    swarm_troughs = pandas.DataFrame(index=np.arange(T), columns=columns)
    swarm_troughs['sat'] = 'A'
    swarm_troughs['seg_e1_mlat'] = 45
    swarm_troughs['seg_e2_mlat'] = 75
    swarm_troughs['seg_e1_mlt'] = 0
    swarm_troughs['seg_e2_mlt'] = 0
    swarm_troughs['tec_ind'] = np.arange(T)

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


def test_get_trough_ratios(trough_setup):
    tec_troughs, swarm_troughs, mlat_grid, mlt_grid = trough_setup
    ta_trough, ta_total, sl_trough, sl_total = compare.get_trough_ratios(tec_troughs, swarm_troughs)
    assert ta_total == tec_troughs.size
    assert ta_trough == 2 * 20 + 2 * 20 + 4 * 20 + 2 * 10 + 2 * 20
    assert sl_trough / sl_total == pytest.approx(28 / 210)


def test_compare(trough_setup):
    tec_troughs, swarm_troughs, mlat_grid, mlt_grid = trough_setup
    ssmlon = np.zeros(tec_troughs.shape[0])
    times = np.datetime64("2000-01-01T00:00:00") + np.arange(tec_troughs.shape[0]) * np.timedelta64(1, 'h')
    results = compare.compare(times, tec_troughs, swarm_troughs, ssmlon, mlat_grid, mlt_grid)
    assert np.all(results['swarm_trough'] == [True, True, True, True, True, False, False])
    assert np.all(results['tec_trough'] == [True, True, True, False, False, True, False])
    assert np.all(results['mlon'] == 180)
    assert results['tec_ewall'][0] == results['swarm_ewall'][0]
    assert results['tec_pwall'][0] == results['swarm_pwall'][0]
    assert results['tec_ewall'][1] > results['swarm_ewall'][1]
    assert results['tec_pwall'][1] < results['swarm_pwall'][1]
    assert results['tec_ewall'][2] < results['swarm_ewall'][2]
    assert results['tec_pwall'][2] > results['swarm_pwall'][2]


def test_process_results(trough_setup):
    tec_troughs, swarm_troughs, mlat_grid, mlt_grid = trough_setup
    ssmlon = (np.arange(tec_troughs.shape[0]) * 90) % 360
    times = np.datetime64("2000-01-01T00:00:00") + np.arange(tec_troughs.shape[0]) * np.timedelta64(1, 'h')
    results = compare.compare(times, tec_troughs, swarm_troughs, ssmlon, mlat_grid, mlt_grid)
    statistics = compare.process_results(results, good_mlon_range=[175, 185])
    statistics = compare.process_results(results, bad_mlon_range=[175, 185])


def test_get_diffs(trough_setup):
    pass


def test_run_single_day():
    pass


def test_run_n_random_days():
    pass


def test_get_random_hyperparams():
    pass


def test_random_parameter_search():
    pass
