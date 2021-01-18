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
    arr = np.arange(21)
    arr[::4] = 0

    r = utils.centered_bn_func(bn.move_min, arr, 5)
    assert np.all(r == 0)
    assert r.shape[0] == 17

    r = utils.centered_bn_func(bn.move_min, arr, 1)
    assert np.all(r == arr)

    with pytest.raises(Exception):
        r = utils.centered_bn_func(bn.move_min, arr, 4)

    r = utils.centered_bn_func(bn.move_min, arr, 5, pad=True)
    assert np.all(r == 0)
    assert r.shape[0] == 21


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
    x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))
    x = np.array([-.7, -.5, 0, 1, 1.1, 1.9, 9.1, 9.9])
    y = np.array([1, 1, 1, 1, 1, 1, 1, 9])
    xi, yi = utils.get_grid_coords(x, y, x_grid, y_grid)
    assert np.all(xi == [-1, 0, 0, 1, 1, 2, 9, 10])
    assert np.all(yi == [1, 1, 1, 1, 1, 1, 1, 9])


def test_get_grid_slice_line():
    """axis aligned has correct length, proper wrapping, proper treatment of 2d v 3d"""
    x_grid, y_grid = np.meshgrid(np.arange(10), np.arange(10))
    dgrid_1 = np.arange(10)[None, :] + np.arange(10)[:, None]
    dgrid_2 = np.ones_like(x_grid)[None, :, :] * np.arange(3)[:, None, None]
    starts = [(1, 1), (1, 1), (7, 1)]
    ends = [(1, 5), (12, 1), (12, 5)]
    x1 = np.array([s[0] for s in starts])
    y1 = np.array([s[1] for s in starts])
    x2 = np.array([s[0] for s in ends])
    y2 = np.array([s[1] for s in ends])
    data_grids = [dgrid_1, dgrid_2]
    results = utils.get_grid_slice_line(x1, y1, x2, y2, data_grids, x_grid, y_grid, linewidth=1)
    assert len(results) == 2
    dprofs1, dprofs2 = results
    assert np.all(dprofs1[0] == [2, 3, 4, 5, 6])
    assert np.all(dprofs1[1] == [2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3])
    assert np.all(dprofs2[0] == [0, 0, 0, 0, 0])
    assert np.all(dprofs2[1] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.all(dprofs2[2] == 2)

    prof, = utils.get_grid_slice_line([0], [1], [5], [5], [np.roll(dgrid_1, 3, 1)], x_grid, y_grid, linewidth=1)
    assert np.allclose(dprofs1[2], prof[0])


def test_average_angles():
    t1 = np.array([np.pi / 6, 2 * np.pi / 3])
    t2 = np.array([11 * np.pi / 6, -2 * np.pi / 3])
    ta = utils.average_angles(t1, t2)
    assert np.allclose(ta, [0, np.pi])
