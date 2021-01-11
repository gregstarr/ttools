import numpy as np
import pytest
import apexpy

from ttools import create_dataset, config, io


map_periods = [np.timedelta64(10, 'm'), np.timedelta64(30, 'm'), np.timedelta64(1, 'h'), np.timedelta64(2, 'h')]


@pytest.fixture
def times():
    yield np.datetime64('2010-01-01T00:00:00') + np.arange(100) * np.timedelta64(5, 'm')


@pytest.mark.parametrize('map_period', map_periods)
def test_assemble_args(times, map_period):
    mlat = np.arange(10)
    mlt = np.arange(10)
    ssmlon = np.random.rand(times.shape[0])
    mlt, mlat = np.meshgrid(mlt, mlat)
    mlat = mlat[None, :, :] * np.ones((times.shape[0], 1, 1))
    mlt = mlt[None, :, :] * np.ones((times.shape[0], 1, 1))
    tec = np.random.rand(*mlat.shape)
    bin_edges = np.arange(-.5, 10)
    bins = [bin_edges, bin_edges]
    args = create_dataset.assemble_binning_args(mlat, mlt, tec, times, ssmlon, bins, map_period)
    assert len(args) == np.ceil((times[-1] - times[0]) / map_period)
    assert args[0][3][0] == times[0]
    assert args[-1][3][0] + map_period >= times[-1]
    assert args[-1][3][0] < times[-1]
    assert args[-1][3][-1] == times[-1]
    for i in range(len(args) - 1):
        assert args[i][3][-1] == args[i + 1][3][0] - np.timedelta64(5, 'm')


@pytest.mark.parametrize('map_period', map_periods)
def test_process_month(madrigal_data_dir, map_period):
    """not that good of a test: wait for bugs and add asserts
    """
    month = np.datetime64('2012-06')
    converter = apexpy.Apex()
    mlat, mlon = create_dataset.get_mag_grid(config.madrigal_lat, config.madrigal_lon, converter)
    bin_edges = np.arange(-.5, 10)
    bins = [bin_edges + 30, bin_edges]
    times, tec, ssmlon, n, std = create_dataset.process_month(month, mlat, mlon, converter, bins, map_period,
                                                              madrigal_data_dir)
    assert times.shape[0] == tec.shape[0] == n.shape[0] == std.shape[0] == ssmlon.shape[0]
    assert np.isnan(tec[times < np.datetime64('2012-06-10')]).all()
    assert np.isnan(tec[times >= np.datetime64('2012-06-11')]).all()
    assert np.isfinite(tec[(times >= np.datetime64('2012-06-10')) * (times < np.datetime64('2012-06-11'))]).any()
    assert not np.isnan(tec).all(axis=(0, 1)).any()
    assert not np.isnan(tec).all(axis=(0, 2)).any()
