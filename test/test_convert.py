"""
- convert with grid converter and slow original converter and verify same results
- verify that SWARM mlat, mlon and mlt are the same as apexpy
"""
import numpy as np
import apexpy

from ttools import convert, io, utils


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


def test_swarm_convert(swarm_data_fn):
    data = io.open_swarm_file(swarm_data_fn)
    converter = apexpy.Apex()
    qdlat, qdlon = converter.convert(data['Latitude'], data['Longitude'], 'geo', 'qd', data['Height'])
    assert np.allclose(qdlat, data['Diplat'])
    assert np.allclose(qdlon, data['Diplon'])
