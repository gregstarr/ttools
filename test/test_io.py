import numpy as np
import os
import apexpy
import tempfile
import yaml
import h5py

from ttools import io, config, utils, convert


def test_get_madrigal_data(madrigal_data_dir):
    """Test that the correct timestamps are returned, correct data is at those timestamps
    """
    start_date = np.datetime64("2012-06-10T10:10:10")
    end_date = np.datetime64("2012-06-10T10:30:01")
    test_fn = os.path.join(madrigal_data_dir, 'gps120610g.001.hdf5')
    file_tec, file_times, lat, lon = io.open_madrigal_file(test_fn)
    tec, times = io.get_madrigal_data(start_date, end_date)
    correct_times = np.datetime64("2012-06-10T10:15:00") + np.arange(4) * np.timedelta64(5, 'm')
    assert np.all(times.astype('datetime64[s]') == correct_times)
    file_tec_r = np.moveaxis(file_tec[:, :, np.in1d(file_times, times.astype('datetime64[s]').astype(int))], -1, 0)
    file_tec_mask = np.isfinite(file_tec_r)
    tec_mask = np.isfinite(tec)
    assert np.all(file_tec_r[file_tec_mask] == tec[tec_mask])


def test_get_madrigal_fill():
    """Verify that empty timestamps are created where no data exists
    """
    start_date = np.datetime64("2012-06-10T22:10:10")
    end_date = np.datetime64("2012-06-11T02:00:00")
    tec, times = io.get_madrigal_data(start_date, end_date)
    correct_times = np.datetime64("2012-06-10T22:15:00") + np.arange(45) * np.timedelta64(5, 'm')
    assert np.all(times.astype('datetime64[s]') == correct_times)
    assert np.all(np.isnan(tec[21:]))


def test_get_madrigal_missing(madrigal_data_dir):
    """Verify that missing latitudes are filled with nans
    """
    start_date = np.datetime64("2012-03-07T10:00:00")
    end_date = np.datetime64("2012-03-07T10:30:00")
    test_fn = os.path.join(madrigal_data_dir, 'gps120307g.002.hdf5')
    *_, lat, lon = io.open_madrigal_file(test_fn)
    tec, _ = io.get_madrigal_data(start_date, end_date)
    missing_lat_mask = ~np.isin(config.madrigal_lat, lat)
    assert np.isnan(tec[:, missing_lat_mask]).all()


def test_get_swarm_data():
    start_date = np.datetime64("2019-12-26T10:10:10.001")
    end_date = np.datetime64("2019-12-26T10:30:01.000")
    data, times = io.get_swarm_data(start_date, end_date, 'C')
    correct_times = np.arange(np.datetime64("2019-12-26T10:10:10.5"), end_date, np.timedelta64(500, 'ms'))
    assert np.all(times == correct_times)

    converter = apexpy.Apex(utils.datetime64_to_datetime(times[0]))
    fin_mask = np.isfinite(data['n'])
    apex_lat, apex_lon = converter.convert(data['lat'][fin_mask], data['lon'][fin_mask], 'geo', 'apex',
                                           data['Height'][fin_mask])
    assert np.allclose(apex_lat, data['apex_lat'][fin_mask])
    assert np.allclose(apex_lon, data['apex_lon'][fin_mask])


def test_get_swarm_fill():
    start_date = np.datetime64("2019-12-26T22:22:22.001")
    end_date = np.datetime64("2019-12-27T02:00:00.000")
    data, times = io.get_swarm_data(start_date, end_date, 'C')
    correct_times = np.arange(np.datetime64("2019-12-26T22:22:22.5"), end_date, np.timedelta64(500, 'ms'))
    no_data_time = np.datetime64("2019-12-27T00:00:00.000")
    assert np.all(times == correct_times)
    assert np.all(np.isnan(data['n'][times >= no_data_time]))


def test_get_tec_data():
    """Test that the correct timestamps are returned, correct data is at those timestamps, correct subsolar point mlon
    """
    start_date = np.datetime64("2012-06-10T10:10:10")
    end_date = np.datetime64("2012-06-10T20:30:01")
    tec, times, ssmlon, n = io.get_tec_data(start_date, end_date)
    correct_times = np.datetime64("2012-06-10T11:00:00") + np.arange(10) * np.timedelta64(1, 'h')
    assert np.all(times.astype('datetime64[s]') == correct_times)
    converter = apexpy.Apex(utils.datetime64_to_datetime(times[0]))
    sslat, sslon = convert.subsol_array(correct_times)
    true_ssmlat, true_ssmlon = converter.geo2apex(sslat, sslon, 50*6371)
    assert abs(true_ssmlon - ssmlon).max() < .1


def test_get_arb_data():
    pass


def test_write_h5():
    data1 = np.random.rand(10)
    data2 = np.random.rand(10)
    data3 = np.random.rand(10, 10, 10)
    with tempfile.TemporaryDirectory() as tmpdr:
        fn = os.path.join(tmpdr, "test.yml")
        io.write_h5(fn, data1=data1, data2=data2, data3=data3)
        assert os.path.exists(fn)
        with h5py.File(fn) as f:
            assert np.all(f['data1'][()] == data1)
            assert np.all(f['data2'][()] == data2)
            assert np.all(f['data3'][()] == data3)


def test_write_yaml():
    with tempfile.TemporaryDirectory() as tmpdr:
        fn = os.path.join(tmpdr, "test.yml")
        io.write_yaml(fn, val_1=1, val_2='test', val_3=[1, 2, 3])
        assert os.path.exists(fn)
        with open(fn) as f:
            out = yaml.safe_load(f)
        assert out['val_1'] == 1
        assert out['val_2'] == 'test'
        assert out['val_3'] == [1, 2, 3]

