import numpy as np
import pytest
import bottleneck as bn

from ttools import utils


@pytest.fixture()
def dt64():
    yield np.datetime64('2000-01-01T00:00:00') + np.arange(10) * np.timedelta64(1, 'h')


def test_datetime64_to_timestamp(dt64):
    """https://www.epochconverter.com/
    """
    ts = utils.datetime64_to_timestamp(dt64)
    assert np.all(ts == 946684800 + np.arange(10) * 60 * 60)


def test_datetime64_to_datetime(dt64):
    dt = utils.datetime64_to_datetime(dt64)
    assert all([d.year == 2000 for d in dt])
    assert all([d.month == 1 for d in dt])
    assert all([d.day == 1 for d in dt])
    assert all([d.hour == i for i, d in enumerate(dt)])
    assert all([d.minute == 0 for d in dt])
    assert all([d.second == 0 for d in dt])


def test_centered_bn_func():
    """even, odd, width=1, pad"""
    arr = np.arange(20)
    arr[::4] = 0

    r = utils.centered_bn_func(bn.move_min, arr, 5)
    assert np.all(r == 0)
    assert r.shape[0] == 16

    r = utils.centered_bn_func(bn.move_min, arr, 1)
    assert np.all(r == arr)

    with pytest.raises(Exception):
        r = utils.centered_bn_func(bn.move_min, arr, 4)

    r = utils.centered_bn_func(bn.move_min, arr, 5, pad=True)
    assert np.all(r == 0)
    assert r.shape[0] == 20


def test_moving_func_trim():
    """even, odd, width=1"""
    arr = np.arange(20)
    arr[::4] = 0

    r1, r2, r3 = utils.moving_func_trim(5, arr, arr, arr)
    assert np.all(r1 == arr[2:-2])
    assert np.all(r2 == arr[2:-2])
    assert np.all(r3 == arr[2:-2])

    r1, r2, r3 = utils.moving_func_trim(1, arr, arr, arr)
    assert np.all(r1 == arr)
    assert np.all(r2 == arr)
    assert np.all(r3 == arr)

    with pytest.raises(Exception):
        r1, r2, r3 = utils.moving_func_trim(4, arr, arr, arr)


def test_get_grid_coords():
    pass


def test_average_angles():
    pass


def test_get_grid_slice_line():
    pass