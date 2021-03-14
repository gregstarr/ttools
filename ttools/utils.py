import numpy as np
import datetime
import os
from skimage.measure import profile_line


def datetime64_to_timestamp(dt64):
    """Convert single / array of numpy.datetime64 to timestamps (seconds since epoch)

    Parameters
    ----------
    dt64: numpy.ndarray[datetime64]

    Returns
    -------
    timestamp: numpy.ndarray[float]
    """
    return (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')


def datetime64_to_datetime(dt64):
    """Convert single datetime64 to datetime

    Parameters
    ----------
    dt64: numpy.ndarray[datetime64]

    Returns
    -------
    list[datetime]
    """
    ts = datetime64_to_timestamp(dt64)
    if isinstance(ts, np.ndarray):
        return [datetime.datetime.utcfromtimestamp(t) for t in ts]
    return datetime.datetime.utcfromtimestamp(ts)


def decompose_datetime64(dt64):
    """Convert array of np.datetime64 to an array (N x 3) of year, month (jan=1), day (1 index)

    Parameters
    ----------
    dt64: numpy.ndarray[datetime64]

    Returns
    -------
    idx: numpy.ndarray (N x 3)
    """
    year_floor = dt64.astype('datetime64[Y]')
    month_floor = dt64.astype('datetime64[M]')

    year = year_floor.astype(int) + 1970
    month = (dt64.astype('datetime64[M]') - year_floor).astype(int) + 1
    day = (dt64.astype('datetime64[D]') - month_floor).astype(int) + 1

    return np.column_stack((year, month, day))


def no_ext_fn(fn):
    """return name of file with no path or extension

    Parameters
    ----------
    fn: str

    Returns
    -------
    str
    """
    return os.path.splitext(os.path.basename(fn))[0]


def centered_bn_func(func, arr, window_diameter, pad=False, **kwargs):
    """Call a centered bottleneck moving window function on an array, optionally padding with the edge values to keep
    the same shape. Window moves through axis 0.

    Parameters
    ----------
    func: bottleneck moving window function
    arr: numpy.ndarray
    window_diameter: int
        odd number window width
    pad: bool
        whether to pad to keep same shape or not
    kwargs
        passed to func

    Returns
    -------
    numpy.ndarray
    """
    window_radius = window_diameter // 2
    assert (2 * window_radius + 1) == window_diameter, "window_diameter must be odd"
    if pad:
        arr = np.pad(arr, window_radius, mode='edge')
    return func(arr, window_diameter, **kwargs)[2 * window_radius:]


def moving_func_trim(window_diameter, *arrays):
    """Trim any number of arrays to valid dimension after calling a centered bottleneck moving window function

    Parameters
    ----------
    window_diameter: int
        odd number window width
    arrays: 1 or more numpy.ndarray

    Returns
    -------
    tuple of numpy.ndarrays
    """
    window_radius = window_diameter // 2
    assert (2 * window_radius + 1) == window_diameter, "window_diameter must be odd"
    if window_radius == 0:
        return (array for array in arrays)
    return (array[window_radius:-window_radius] for array in arrays)


def get_grid_coords(x, y, x_grid, y_grid):
    """Gets grid indices of points.

    x[0] ~ x_grid[{any}, xi[0]]

    Parameters
    ----------
    x, y: numpy.ndarray[float]
        points to get grid indices of
    x_grid, y_grid: numpy.ndarray[float]
        x and y grids, x_grid varies along axis=1, y_grid varies along axis=0
    Returns
    -------
    xi, yi: numpy.ndarray[int]
    """
    y0 = y_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]
    x0 = x_grid[0, 0]
    dx = x_grid[0, 1] - x_grid[0, 0]
    yi = np.round((y - y0) / dy).astype(int)
    xi = np.round((x - x0) / dx).astype(int)
    return xi, yi


def average_angles(theta1, theta2):
    """Average two angles together by averaging in cartesian space and converting back to polar.

    Parameters
    ----------
    theta1, theta2: numpy.ndarray[float]
        angles to average together

    Returns
    -------
    numpy.ndarray
        averaged angles
    """
    x1 = np.cos(theta1)
    y1 = np.sin(theta1)
    x2 = np.cos(theta2)
    y2 = np.sin(theta2)
    x_avg = (x1 + x2) / 2
    y_avg = (y1 + y2) / 2
    theta_avg = np.arctan2(y_avg, x_avg)
    return theta_avg


def get_grid_slice_line(x1, y1, x2, y2, data_grids, x_grid, y_grid, **profile_line_kwargs):
    """Get slice of 2D data grids along lines defined by endpoints.

    Parameters
    ----------
    x1, y1, x2, y2: numpy.ndarray[float]
        line endpoints
    data_grids: array-like
        list of 2D data grids to slice, can also be 3D, first axis corresponding to different lines
    x_grid, y_grid: numpy.ndarray[float]
        coordinate grids
    **profile_line_kwargs: passed to `skimage.measure.profile_line`

    Returns
    -------
    list of lists
        [slices_of_data_grid_1, ..., slices_of_data_grid_n]
    """
    kwargs = dict(linewidth=3, mode='grid-wrap', reduce_func=np.nanmean)
    kwargs.update(profile_line_kwargs)

    xi1, yi1 = get_grid_coords(x1, y1, x_grid, y_grid)
    xi2, yi2 = get_grid_coords(x2, y2, x_grid, y_grid)

    xi2[abs(xi1 - xi2) > x_grid.shape[1] // 2] += x_grid.shape[1]

    results = [[] for _ in data_grids]
    for t in range(len(xi1)):
        src = (yi1[t], xi1[t])
        dst = (yi2[t], xi2[t])
        for i, dgrid in enumerate(data_grids):
            if len(dgrid.shape) == 3:
                dprof = profile_line(dgrid[t].astype(float), src, dst, **kwargs)
            else:
                dprof = profile_line(dgrid.astype(float), src, dst, **kwargs)
            results[i].append(dprof)
    return results


def polar_to_cart(lat, lon, period=24):
    r = 90 - lat
    t = lon * 2 * np.pi / period
    return r * np.cos(t), r * np.sin(t)
