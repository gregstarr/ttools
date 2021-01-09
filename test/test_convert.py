import numpy as np
import apexpy

from ttools import convert, io, utils


def test_subsolar():
    times = np.datetime64("2000-01-01T00:00:00") + np.random.rand(1000) * 25 * 365 * 24 * np.timedelta64(1, 'h')
    print(times.min(), times.max())
    times_dt = utils.datetime64_to_datetime(times)
    lat, lon = convert.subsol_array(times)
    for i, t in enumerate(times_dt):
        true_lat, true_lon = apexpy.helpers.subsol(t)
        assert true_lat == lat[i]
        assert true_lon == lon[i]


def test_mlon_to_mlt():
    N = 100
    lat = (np.random.rand(N) - .5) * 90
    lon = (np.random.rand(N) - .5) * 180
    times = np.datetime64("2000-01-01T00:00:00") + np.random.rand(N) * 25 * 365 * 24 * np.timedelta64(1, 'h')
    times_dt = utils.datetime64_to_datetime(times)
    converter = apexpy.Apex(utils.datetime64_to_datetime(times[0]))
    apex_lat, apex_lon = converter.geo2apex(lat, lon, 100)
    qd_lat, qd_lon = converter.geo2qd(lat, lon, 100)
    assert np.all(apex_lon == qd_lon)
    mlt = convert.mlon_to_mlt_array(apex_lon, times, converter)
    for i, t in enumerate(times_dt):
        true_lat, true_mlt = converter.convert(lat[i], lon[i], 'geo', 'mlt', 100, datetime=t)
        assert true_lat == apex_lat[i]
        assert true_mlt == mlt[i]


def test_geo_to_mlt_grid():
    lat = np.arange(10) + 30
    lon = np.arange(10) + 30
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    times = np.datetime64("2012-10-10T10:10:10") + np.arange(10) * np.timedelta64(1, 'h')
    mlat, mlt = convert.geo_to_mlt_grid(lat, lon, times, height=100)

    converter = apexpy.Apex()
    for i, t in enumerate(times):
        true_mlat, true_mlt = converter.convert(lat_grid.ravel(), lon_grid.ravel(), 'geo', 'mlt', height=100,
                                               datetime=utils.datetime64_to_datetime(t))
        true_mlat = true_mlat.reshape(lat_grid.shape)
        true_mlt = true_mlt.reshape(lat_grid.shape)
        assert np.allclose(mlat[i], true_mlat)
        assert np.allclose(mlt[i], true_mlt)
