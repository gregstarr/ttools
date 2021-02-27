import numpy as np
import pytest
import apexpy
import tempfile
import os
import h5py

from ttools import create_dataset, config, io, utils

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
def test_process_file(madrigal_data_dir, map_period):
    """not that good of a test: wait for bugs and add asserts
    """
    start_date = np.datetime64('2012-06-08')
    end_date = np.datetime64('2012-06-13')
    converter = apexpy.Apex()
    mlat, mlon = create_dataset.get_mag_grid(config.madrigal_lat, config.madrigal_lon, converter)
    bin_edges = np.arange(-.5, 10)
    bins = [bin_edges + 30, bin_edges]
    times, tec, ssmlon, n, std = create_dataset.process_file(start_date, end_date, mlat, mlon, converter, bins,
                                                             map_period, madrigal_data_dir)
    assert times.shape[0] == tec.shape[0] == n.shape[0] == std.shape[0] == ssmlon.shape[0]
    assert np.isnan(tec[times < np.datetime64('2012-06-10')]).all()
    assert np.isnan(tec[times >= np.datetime64('2012-06-11')]).all()
    assert np.isfinite(tec[(times >= np.datetime64('2012-06-10')) * (times < np.datetime64('2012-06-11'))]).any()
    assert not np.isnan(tec).all(axis=(0, 1)).any()
    assert not np.isnan(tec).all(axis=(0, 2)).any()


def test_calculate_bins():
    mlat = np.arange(10)[None, :, None] * np.ones((1, 1, 10))
    mlt = np.arange(10)[None, None, :] * np.ones((1, 10, 1))
    tec = np.zeros((1, 10, 10))
    tec[0, 0, 0] = 10
    tec[0, 0, -1] = 20
    tec[0, -1, 0] = 30
    times = ssmlon = np.ones(1) * np.nan
    be = np.array([-.5, 4.5, 9.5])
    bins = [be, be]
    out_t, out_tec, out_ssm, out_n, out_std = create_dataset.calculate_bins(mlat.ravel(), mlt.ravel(), tec.ravel(),
                                                                            times, ssmlon, bins)
    assert np.isnan(out_t)
    assert np.isnan(out_ssm)
    assert out_tec.shape == (2, 2)
    assert out_tec[0, 0] == 10 / 25
    assert out_tec[0, 1] == 20 / 25
    assert out_tec[1, 0] == 30 / 25
    assert out_tec[1, 1] == 0
    assert np.all(out_n == 25)


def test_process_dataset():
    start_date = np.datetime64("2012-03-07")
    end_date = np.datetime64("2012-03-08")
    file_dt = np.timedelta64(12, 'h')
    mlat_bins = np.array([35, 45, 55, 65])
    mlt_bins = np.array([-1.5, -.5, .5, 1.5])

    def fn_pattern(date):
        return f"{date.astype('datetime64[h]')}.h5"

    dates = np.arange(start_date, end_date, file_dt)

    with tempfile.TemporaryDirectory() as tempdir:
        files = [os.path.join(tempdir, fn_pattern(d)) for d in dates]
        create_dataset.process_dataset(start_date, end_date, mlat_bins, mlt_bins, apex_dt=np.timedelta64(365, 'D'),
                                       file_dt=file_dt, output_dir=tempdir, file_name_pattern=fn_pattern)

        grid_fn = os.path.join(tempdir, 'grid.h5')
        assert os.path.exists(grid_fn)
        with h5py.File(grid_fn, 'r') as f:
            mlt_vals = f['mlt'][()]
            mlat_vals = f['mlat'][()]
        assert np.all(mlt_vals == [-1, 0, 1])
        assert np.all(mlat_vals == [40, 50, 60])

        for f, d in zip(files, dates):
            assert os.path.exists(f)
            tec, times, ssmlon, n, std = io.open_tec_file(f)
            assert tec.shape == (12, 3, 3)
            assert utils.datetime64_to_timestamp(d) == times[0]
