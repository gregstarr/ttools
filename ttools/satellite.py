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

SATELLITES = {'swarm': ('A', 'B', 'C'), 'dmsp': ('dmsp15', 'dmsp16', 'dmsp17', 'dmsp18')}
DIRECTIONS = {'up': (45, 75), 'down': (75, 45)}

import numpy as np
import bottleneck as bn
import pandas
from scipy.interpolate import interp1d

from ttools import io, utils


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
    xp = utils.centered_bn_func(bn.move_median, x, 21, pad=True, min_count=5)
    yp = utils.centered_bn_func(bn.move_median, y, 21, pad=True, min_count=5)
    d = np.hypot(x - xp, y - yp)
    bad, = np.nonzero(d > 10 * bn.nanmean(d))
    new_x = xp[bad]
    new_y = yp[bad]
    new_lat = 90 - np.hypot(new_x, new_y)
    new_lon = np.degrees(np.arctan2(new_y, new_x))
    fixed_lon[bad] = new_lon
    fixed_lat[bad] = new_lat
    print(f"Fixed {bad.shape[0]} bad coordinates")
    return fixed_lat, fixed_lon


def process_data_interval(times, ne, extra_data=None, median_window=3, mean_window=481):
    """take log of Ne, perform moving median filter, estimate background using moving average filter

    Parameters
    ----------
    data, times: numpy.ndarray
    median_window, mean_window: int

    Returns
    -------
    times, log_ne, background, mlat, mlt
    """
    if extra_data is None:
        extra_data = []
    ne[ne <= 0] = np.nan  # get rid of zeros
    log_ne = np.log10(ne)  # take log
    log_ne = utils.centered_bn_func(bn.move_median, log_ne, median_window, min_count=1)  # median
    times, *extra_data = utils.moving_func_trim(median_window, times, *extra_data)  # trim
    background = utils.centered_bn_func(bn.move_mean, log_ne, mean_window, min_count=10)  # moving average
    times, log_ne, *extra_data = utils.moving_func_trim(mean_window, times, log_ne, *extra_data)  # trim
    return [times, log_ne, background] + extra_data


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
    if centers.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    best_centers = np.argmin(abs(timestamp[None, centers] - tec_time[:, None]), axis=1)
    return starts[best_centers], ends[best_centers]


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

    dist = (ends[:, None] - starts[None, :]).astype(float)
    dist[dist <= 0] = np.inf
    end_pick_mask = np.isfinite(dist).any(axis=1)
    if end_pick_mask.sum() == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    start_pick_ind, end_pick_ind = np.unique(np.argmin(dist[end_pick_mask, :], axis=1), return_index=True)

    return starts[start_pick_ind], ends[end_pick_mask][end_pick_ind]


def find_troughs_in_segment(mlat, smooth_dne, threshold=-.15, width_min=1, width_max=17, fin_rmin=.25):
    """find troughs in a 45-75 segment. `mlat` contains no NaNs, smooth_dne may contain NaNs.

    Parameters
    ----------
    mlat, smooth_dne: numpy.ndarray[float]
    threshold, width_min, width_max, fin_rmin: float

    Returns
    -------
    trough:
        - if trough: min_idx, edge_1, edge_2
        - if no trough: False
    """
    trough_candidates = []
    # fill NaNs
    fin_mask = np.isfinite(smooth_dne)
    smooth_dne_i = smooth_dne.copy()  # i for interpolated
    if not fin_mask.any():
        return []
    if not fin_mask.all():
        interpolator = interp1d(mlat[fin_mask], smooth_dne[fin_mask], kind='previous', bounds_error=False, fill_value=0)
        smooth_dne_i[~fin_mask] = interpolator(mlat[~fin_mask])
    # find zero crossings
    zerox, = np.nonzero(np.diff(smooth_dne_i >= 0))
    zerox += 1
    zerox = np.concatenate(([0], zerox, [smooth_dne_i.shape[0] - 1]))
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
        if fin_mask[edge_1:edge_2].mean() < fin_rmin:
            continue
        min_idx = edge_1 + np.nanargmin(smooth_dne[edge_1:edge_2])
        dne_min = smooth_dne[min_idx]
        if dne_min <= threshold:
            trough_candidates.append((min_idx, edge_1, edge_2))
    return trough_candidates


def get_segments_data(tec_times, mission, *extra_data_keys, get_data_func=None):
    """This function collects and organizes satellite data into "orbital segments" from 45 - 75 mlat. There are two
    orbital segments per orbit of the satellite: 'up' and 'down', where the mlat is increasing and decreasing with
    time respectively. Finally there are multiple satellites and so for each tec time, there are 2 * n_sats segments.

    Parameters
    ----------
    tec_times: numpy.ndarray[datetime64]
    mission: str

    Returns
    -------
    dict
        {
            sat: [
                dict{
                    'up': [
                        dict{times, mlat, mlt, dne, smooth_dne, direction, tec_time_1},
                        ...,
                        dict{times, mlat, mlt, dne, smooth_dne, direction, tec_time_N},
                    ],
                    'down': [...]
                },
                ...
            ]
        }
    """
    ne_key = 'n'
    if mission.lower() == 'swarm':
        get_data = io.get_swarm_data
        if not extra_data_keys:
            extra_data_keys = ('mlat', 'mlt')
    elif mission.lower() == 'dmsp':
        get_data = io.get_dmsp_data
        ne_key = 'ne'
        if not extra_data_keys:
            extra_data_keys = ('mlat', 'mlt', 'hor_ion_v')
    if get_data_func is not None:
        get_data = get_data_func

    satellites = SATELLITES[mission]

    dt = np.timedelta64(5, 'h')
    i = np.argwhere(abs(np.diff(tec_times)) > 2 * dt)[:, 0]
    interval_start = np.concatenate((tec_times[[0]], tec_times[i + 1]), axis=0) - dt
    interval_end = np.concatenate((tec_times[i], tec_times[[-1]]), axis=0) + dt
    sat_segments = {sat: {direction: [] for direction in DIRECTIONS} for sat in satellites}

    for idx in range(len(interval_start)):
        tt = tec_times[(tec_times > interval_start[idx]) * (tec_times < interval_end[idx])]
        sat_data, sat_times = get_data(interval_start[idx], interval_end[idx])
        for sat in satellites:
            if mission == 'dmsp':
                times, log_ne, background, *extra_data = process_data_interval(sat_times, sat_data[sat][ne_key], [sat_data[sat][key] for key in extra_data_keys], mean_window=321)
            else:
                times, log_ne, background, *extra_data = process_data_interval(sat_times, sat_data[sat][ne_key], [sat_data[sat][key] for key in extra_data_keys])
            if 'hor_ion_v' in extra_data_keys:
                extra_data[-1] = utils.centered_bn_func(bn.move_mean, extra_data[-1], 9, pad=True, min_count=1)
            if 'T_elec' in extra_data_keys:
                extra_data[-1] = utils.centered_bn_func(bn.move_mean, extra_data[-1], 11, pad=True, min_count=1)
            dne = log_ne - background
            smooth_dne = utils.centered_bn_func(bn.move_mean, dne, 9, pad=True, min_count=1)
            fin_mask = np.isfinite(extra_data[0])
            for direction, (enter, exit) in DIRECTIONS.items():
                starts, stops = get_closest_segment(times[fin_mask], extra_data[0][fin_mask], tt, enter, exit)
                for i, (start, stop) in enumerate(zip(starts, stops)):
                    sl = slice(start, stop)
                    segment_data = {
                        'times': times[fin_mask][sl],
                        'log_ne': log_ne[fin_mask][sl],
                        'dne': dne[fin_mask][sl],
                        'smooth_dne': smooth_dne[fin_mask][sl],
                        'direction': direction,
                        'tec_time': tec_times[i],
                    }
                    for i, key in enumerate(extra_data_keys):
                        segment_data[key] = extra_data[i][fin_mask][sl]
                    sat_segments[sat][direction].append(segment_data)
    return sat_segments


def get_troughs(segments):
    sat_troughs = []
    sat_ind = 0
    for sat, sat_segments in segments.items():
        for direction, segms in sat_segments.items():
            for i, seg in enumerate(segms):
                trough_candidates = find_troughs_in_segment(seg['mlat'], seg['smooth_dne'])
                data_rows = []
                seg_info = (sat_ind, sat, seg['mlat'][0], seg['mlt'][0], seg['mlat'][-1],
                            seg['mlt'][-1], i, direction)
                trough_unpacked = (False, 0, 0, 0, 0, 0, 0, 0)
                data_rows.append(trough_unpacked + seg_info)
                for tc in trough_candidates:
                    min_idx, e1_idx, e2_idx = tc
                    trough_unpacked = (True, seg['mlat'][min_idx], seg['mlt'][min_idx], seg['smooth_dne'][min_idx],
                                       seg['mlat'][e1_idx], seg['mlt'][e1_idx], seg['mlat'][e2_idx], seg['mlt'][e2_idx])
                    data_rows.append(trough_unpacked + seg_info)
                sat_troughs += data_rows
                sat_ind += 1
    sat_troughs = pandas.DataFrame(data=sat_troughs,
                                   columns=['trough', 'min_mlat', 'min_mlt', 'min_dne', 'e1_mlat', 'e1_mlt', 'e2_mlat',
                                            'e2_mlt', 'sat_ind', 'sat', 'seg_e1_mlat', 'seg_e1_mlt', 'seg_e2_mlat',
                                            'seg_e2_mlt', 'tec_ind', 'direction'])
    assert not np.any(np.isnan(sat_troughs[['min_mlat', 'min_mlt', 'min_dne', 'e1_mlat', 'e1_mlt', 'e2_mlat', 'e2_mlt',
                                            'sat_ind', 'seg_e1_mlat', 'seg_e1_mlt', 'seg_e2_mlat', 'seg_e2_mlt',
                                            'tec_ind']]))
    return sat_troughs


def fix_trough_list(troughs):
    repeats = np.argwhere((troughs['sat_ind'].values[:-1] == 143) & (troughs['sat_ind'].values[1:] == 0))[:, 0] + 1
    tec_ind = troughs['tec_ind'].values
    sat_ind = troughs['sat_ind'].values
    for r in repeats:
        tec_ind[r:] += 24
        sat_ind[r:] += 144
    troughs['tec_ind'] = tec_ind
    troughs['sat_ind'] = sat_ind
    return troughs
