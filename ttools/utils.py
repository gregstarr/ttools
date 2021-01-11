import numpy as np
import datetime
import os
import bottleneck as bn
from skimage.util import view_as_blocks


def datetime64_to_timestamp(dt64):
    """Convert single / array of numpy.datetime64 to timestamps (seconds since epoch)

    Parameters
    ----------
    dt64: numpy.ndarray[datetime64]

    Returns
    -------
    timestamp: numpy.ndarray[float]
    """
    return (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


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


def idx_to_timestamp(idx):
    """Convert (N x 3) array of year, month (jan = 1), hour to timestamps

    Parameters
    ----------
    idx: numpy.ndarray (N x 3)

    Returns
    -------
    timestamps: numpy.ndarray (N, )
    """
    return idx_to_datetime64(idx).astype(int)


def idx_to_datetime64(idx):
    """Convert (N x 3) array of year, month (jan = 1), hour to numpy.datetime64

    Parameters
    ----------
    idx: numpy.ndarray (N x 3)

    Returns
    -------
    datetime64: numpy.ndarray[datetime64]
    """
    year = (idx[:, 0] - 1970).astype('datetime64[Y]')
    month = (idx[:, 1] - 1).astype('timedelta64[M]')
    hour = idx[:, 2].astype('timedelta64[h]')
    npdt = year + month + hour
    return npdt.astype('datetime64[s]')


def datetime64_to_idx(dt64):
    """Convert array of np.datetime64 to an array (N x 3) of year, month (jan = 1), hour

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
    month = (dt64 - year_floor).astype('timedelta[M]').astype(int) + 1
    hour = (dt64 - month_floor).astype('timedelta[h]').astype(int)

    return np.column_stack((year, month, hour))


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


def get_random_map_id(start_time=np.datetime64("2013-12-03T00:00:00"), end_time=np.datetime64("2019-12-30T00:00:00")):
    """gets random year, month and hour within the time range

    Parameters
    ----------
    start_time: np.datetime64
    end_time: np.datetime64

    Returns
    -------
    year, month, hour: (int, int, int)
    """
    dset_range = (end_time.astype('datetime64[h]') - start_time.astype('datetime64[h]')).astype(int)
    hours_offset = np.random.randint(0, dset_range)
    map_time = start_time + np.timedelta64(hours_offset, 'h')
    year = map_time.astype('datetime64[Y]').astype(int) + 1970
    month = map_time.astype('datetime64[M]').astype(int) % 12 + 1
    index = (map_time.astype('datetime64[h]') - map_time.astype('datetime64[M]')).astype(int)
    return year, month, index


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
    window_radius = window_diameter // 2
    if pad:
        arr = np.pad(arr, window_radius, mode='edge')
    return func(arr, window_diameter, **kwargs)[2 * window_radius:]


def moving_func_trim(window_diameter, *arrays):
    window_radius = window_diameter // 2
    if len(arrays) == 1 and isinstance(arrays[0], dict):
        return {k: v[window_radius:-window_radius] for k, v in arrays[0].items()}
    return (array[window_radius:-window_radius] for array in arrays)
