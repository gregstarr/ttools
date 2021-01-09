"""
SWARM Trough detection:
    - 3 point median filter
    - cut into 45-75 MLAT segments
    - background = sliding window of 480 points
    - check detrended logarithmic density to see if it has negative peak that both corresponds to the local Ne
        minimum within "the window" and lower than a threshold of -0.3
    - mark poleward and equatorward transitions back to 0 as the walls
    - filter out troughs smaller than 1 degree wide and larger than 18 degrees wide
    - if more than one trough is identified in a segment, choose the equatorward one

Terms:
    - segment: "orbital segment" from Aa 2020, section of data from 45 - 75 mlat
    - orbit: once around the globe by the satellite
    - interval: section of data surrounding a tec map (default is 2 hours on either side, total of 5 hours)

Example:
```
data, times = io.get_swarm_data(start_time, end_time, sat)
times, log_ne, background, mlat, mlt = swarm.process_swarm_data_interval(data, times)
segment = swarm.get_closest_segment(times, mlat, tec_time, 45, 75)
dne = log_ne - background
smooth_dne = utils.centered_bn_func(bn.move_mean, dne, 10, pad=True, min_count=1)
trough = swarm.find_troughs_in_segment(mlat[segment], smooth_dne[segment])
if trough:
    min_idx, edge_1, edge_2 = trough
    ...
```
"""
import numpy as np
import bottleneck as bn
from scipy.signal import convolve2d
from scipy.stats import binned_statistic_2d
from scipy.interpolate import interp1d

from ttools import io, utils


SWARM_SATELLITES = ('A', 'B', 'C')
SWARM_LOW_SATELLITE_PAIR = ('A', 'C')
SWARM_HIGH_SATTELITE = 'B'


def fix_latlon(lat, lon):
    """Fix errors arising from using a moving average filter on longitude near the 180/-180 crossover.

    Parameters
    ----------
    lat: numpy.ndarray[float]
    lon: numpy.ndarray[float]

    Returns
    -------
    fixed_lat, fixed_lon: numpy.ndarray[float]
    """
    fixed_lat = lat.copy()
    fixed_lon = lon.copy()

    theta = np.radians(lon)
    r = 90 - lat
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xp = utils.centered_bn_func(bn.move_median, x, 20, pad=True, min_count=5)
    yp = utils.centered_bn_func(bn.move_median, y, 20, pad=True, min_count=5)
    d = np.hypot(x - xp, y - yp)
    bad, = np.nonzero(d > 10 * bn.nanmean(d))
    new_x = xp[bad]
    new_y = yp[bad]
    new_lat = 90 - np.hypot(new_x, new_y)
    new_lon = np.degrees(np.arctan2(new_y, new_x))
    fixed_lon[bad] = new_lon
    fixed_lat[bad] = new_lat
    return fixed_lat, fixed_lon


def get_pixel_mask(mlt, mlat, radius):
    bins = [np.arange(29.5, 90), np.arange(-12 / 360, 24, 24 / 360)]
    path = binned_statistic_2d(mlat, mlt, np.ones(mlat.shape[0]), 'sum', bins=bins).statistic
    kernel = np.ones(radius + 1)[None, :]
    m = convolve2d(path, kernel, mode='same', boundary='wrap')
    return m > 0


def process_swarm_data_interval(data, times, median_window=3, mean_window=481):
    """take log of Ne, perform moving median filter, estimate background using moving average filter

    Parameters
    ----------
    data, times: numpy.ndarray
    median_window, mean_window: int

    Returns
    -------
    times, log_ne, background, mlat, mlt
    """
    ne = data['n']
    ne[ne <= 0] = np.nan  # get rid of zeros
    log_ne = np.log10(ne)  # take log
    log_ne = utils.centered_bn_func(bn.move_median, log_ne, median_window, min_count=1)  # median
    times, mlat, mlt = utils.moving_func_trim(median_window, times, data['apex_lat'], data['mlt'])  # trim
    background = utils.centered_bn_func(bn.move_mean, log_ne, mean_window, min_count=10)  # moving average
    times, log_ne, mlat, mlt = utils.moving_func_trim(mean_window, times, log_ne, mlat, mlt)  # trim
    return times, log_ne, background, mlat, mlt


def get_closest_segment(timestamp, mlat, tec_time, enter_lat, exit_lat=None):
    """given a time corresponding to a TEC map, find the closest segment defined by an enter and exit latitude

    Parameters
    ----------
    timestamp, mlat: numpy.ndarray
    tec_time, enter_lat: float
    exit_lat: (optional) float

    Returns
    -------
    slice
    """
    if exit_lat is None:
        enter_mask = mlat >= enter_lat
        exit_mask = mlat < enter_lat
    else:
        if enter_lat < exit_lat:
            enter_mask = mlat >= enter_lat
            exit_mask = mlat >= exit_lat
        else:
            enter_mask = mlat <= enter_lat
            exit_mask = mlat < exit_lat
    starts, ends = get_region_bounds(enter_mask, exit_mask)
    centers = (starts + ends) // 2
    best_center = np.argmin(abs(timestamp[centers] - tec_time))
    return slice(starts[best_center], ends[best_center])


def get_region_bounds(enter_mask, exit_mask):
    """Given masks indicating where the region has started and where the region has ended, return starting and ending
    indices for the region. For finding starting and ending indices of 45-75 MLat segments.

    example:
        enter_mask = mlat >= 45
        exit_mask = mlat > 75

    Parameters
    ----------
    enter_mask: numpy.ndarray[bool]
    exit_mask: numpy.ndarray[bool]

    Returns
    -------
    starts, ends: numpy.ndarray[int]
    """
    # enter_mask = utils.centered_bn_func(bn.move_median, enter_mask, 7, pad=True, min_count=1)
    # exit_mask = utils.centered_bn_func(bn.move_median, exit_mask, 7, pad=True, min_count=1)
    starts, = np.nonzero(np.diff(enter_mask.astype(int)) == 1)
    ends, = np.nonzero(np.diff(exit_mask.astype(int)) == 1)
    starts += 1
    ends += 1

    trimmed_starts = starts.copy()
    trimmed_ends = ends.copy()
    if ends[0] < starts[0]:
        trimmed_ends = trimmed_ends[1:]
    if ends[-1] < starts[-1]:
        trimmed_starts = trimmed_starts[:-1]

    assert trimmed_starts.shape[0] == trimmed_ends.shape[0], "BAD INPUT TO `swarm.get_region_bounds`"

    return trimmed_starts, trimmed_ends


def find_troughs_in_segment(mlat, smooth_dne, threshold=-.2, width_min=1, width_max=17):
    """find troughs in a 45-75 segment. `mlat` contains no NaNs, smooth_dne may contain NaNs.

    Parameters
    ----------
    mlat, smooth_dne: numpy.ndarray[float]
    threshold, width_min, width_max: float

    Returns
    -------
    trough:
        - if trough: min_idx, edge_1, edge_2
    """
    # fill NaNs
    fin_mask = np.isfinite(smooth_dne)
    smooth_dne_i = smooth_dne.copy()  # i for interpolated
    if not fin_mask.all():
        interpolator = interp1d(mlat[fin_mask], smooth_dne[fin_mask], kind='previous')
        smooth_dne_i[~fin_mask] = interpolator(mlat[~fin_mask])
    # find zero crossings
    zerox, = np.nonzero(np.diff(smooth_dne_i >= 0))
    zerox += 1
    # check each interval
    if mlat[0] > mlat[-1]:
        iterator = range(zerox.shape[0] - 2, -1, -1)
    else:
        iterator = range(zerox.shape[0] - 1)
    for i in iterator:
        edge_1 = zerox[i]
        edge_2 = zerox[i + 1]
        width = abs(mlat[edge_1] - mlat[edge_2])
        if not width_min <= width <= width_max:
            continue
        if fin_mask[edge_1:edge_2].mean() < .5:
            continue
        min_idx = edge_1 + np.nanargmin(smooth_dne_i[edge_1:edge_2])
        dne_min = smooth_dne_i[min_idx]
        if dne_min <= threshold:
            # found a trough: return indices of min and walls
            return min_idx, edge_1, edge_2
    return False
