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
"""
import pysatCDF
import os
import glob
import numpy as np
import bottleneck as bn
from scipy.signal import convolve2d
from scipy.stats import binned_statistic_2d


SWARM_SATELLITES = ('A', 'B', 'C')
SWARM_LOW_SATELLITE_PAIR = ('A', 'C')
SWARM_HIGH_SATTELITE = 'B'


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


def get_data_from_file(file, *fields):
    with pysatCDF.CDF(file) as f:
        data = f.data
    if len(fields) == 0:
        fields = data.keys()
    return data['Timestamp'], [data[field] for field in fields]


def get_swarm_file_by_time(datetime, sat, base_dir="E:\\swarm\\extracted"):
    year = datetime.astype('datetime64[Y]').astype(int) + 1970
    month = datetime.astype('datetime64[M]').astype(int) % 12 + 1
    day = (datetime.astype('datetime64[D]') - datetime.astype('datetime64[M]')).astype(int) + 1
    return get_swarm_file(year, month, day, sat, base_dir)


def get_swarm_file(year, month, day, sat, base_dir="E:\\swarm\\extracted"):
    fn_pattern = os.path.join(base_dir, "SW_EXTD_EFI{sat}_LP_HM_{year:04d}{month:02d}{day:02d}*.cdf")
    files = glob.glob(fn_pattern.format(sat=sat, year=year, month=month, day=day))
    if not files:
        return None
    return files[-1]


def get_enter_exit(enter_mask, exit_mask, min_length=50):
    starts, = np.nonzero(np.diff(enter_mask.astype(int)) == 1)
    ends, = np.nonzero(np.diff(exit_mask.astype(int)) == -1)
    starts = starts[np.diff(np.concatenate((starts, [enter_mask.shape[0]]))) > min_length]
    ends = ends[np.diff(np.concatenate((ends, [exit_mask.shape[0]]))) > min_length]
    if ends[0] < starts[0]:
        ends = ends[1:]
    if ends[-1] < starts[-1]:
        starts = starts[:-1]
    if abs(starts.shape[0] - ends.shape[0]) == 1:
        starts_too_long = starts.shape[0] > ends.shape[0]
        too_long = starts if starts_too_long else ends
        right = ends if starts_too_long else starts
        anti_identity = np.arange(too_long.shape[0])[:, None] != np.arange(too_long.shape[0])[None, :]
        keep_ind = np.argwhere(anti_identity)[:, 1].reshape((too_long.shape[0], -1))
        v = abs(too_long[keep_ind] - right).var(axis=1)
        starts = too_long[keep_ind[np.argmin(v)]] if starts_too_long else right
        ends = right if starts_too_long else too_long[keep_ind[np.argmin(v)]]
    elif starts.shape != ends.shape:
        raise Exception("Different number of starts and ends!")
    return starts, ends


def find_troughs_in_segment(mlat, smooth_dne, threshold=-.2):
    # find zero crossings
    zeros, = np.nonzero(np.diff(smooth_dne > 0))
    # check each interval
    if mlat[0] > mlat[-1]:
        iterator = range(zeros.shape[0] - 2, -1, -1)
    else:
        iterator = range(zeros.shape[0] - 1)
    for i in iterator:
        z1 = zeros[i]
        z2 = zeros[i + 1]
        width = abs(mlat[z1] - mlat[z2])
        if not 1 <= width <= 17:
            continue
        min_idx = z1 + np.nanargmin(smooth_dne[z1:z2])
        dne_min = smooth_dne[min_idx]
        if dne_min <= threshold:
            # found a trough: return indices of min and walls
            return min_idx, z1, z2
    return False


def fix_mlt_mlat(mlt, mlat):
    theta = np.pi * mlt / 12
    r = 90 - mlat
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    mask = mlat >= 30
    starts, ends = get_enter_exit(mask, mask)
    for s, e in zip(starts, ends):
        sl = slice(s, e)
        xp = centered_bn_func(bn.move_median, x[sl], 20, pad=True, min_count=5)
        yp = centered_bn_func(bn.move_median, y[sl], 20, pad=True, min_count=5)
        d = np.hypot(x[sl] - xp, y[sl] - yp)
        bad, = np.nonzero(d > 10 * bn.nanmean(d))
        new_x = xp[bad]
        new_y = yp[bad]
        new_mlat = 90 - np.hypot(new_x, new_y)
        new_mlt = np.arctan2(new_y, new_x) * 12 / np.pi
        mlt[s + bad] = new_mlt
        mlat[s + bad] = new_mlat
    return mlt, mlat


def process_swarm_data_interval(year, month, index, sat, median_window=3, mean_window=481):
    timestamp, [ne, mlat, mlt] = get_swarm_data_interval(year, month, index, sat)  # get large interval
    mlt, mlat = fix_mlt_mlat(mlt, mlat)
    ne[ne <= 0] = np.nan  # get rid of zeros
    ne = np.log10(ne)  # take log
    ne = centered_bn_func(bn.move_median, ne, median_window, min_count=1)  # median
    timestamp, mlat, mlt = moving_func_trim(median_window, timestamp, mlat, mlt)  # trim
    background = centered_bn_func(bn.move_mean, ne, mean_window, min_count=10)  # moving average
    timestamp, ne, mlat, mlt = moving_func_trim(mean_window, timestamp, ne, mlat, mlt)  # trim
    return timestamp, ne, background, mlat, mlt


def get_closest_segment(timestamp, mlat, tec_time, enter_lat, exit_lat=None):
    if exit_lat is None:
        enter_mask = exit_mask = mlat >= enter_lat
    elif enter_lat < exit_lat:
        enter_mask = mlat >= enter_lat
        exit_mask = mlat <= exit_lat
    else:
        enter_mask = mlat <= enter_lat
        exit_mask = mlat >= exit_lat
    starts, ends = get_enter_exit(enter_mask, exit_mask)
    centers = (starts + ends) // 2
    best_center = np.argmin(abs(timestamp[centers] - tec_time))
    return slice(starts[best_center], ends[best_center])


def get_swarm_data_interval(year, month, index, sat, interval_radius=np.timedelta64(2, 'h'),
                            fields=('n', 'MLat', 'MLT')):
    day = index // 24 + 1
    hour = index % 24
    tec_time = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00")
    start_time = tec_time - interval_radius
    start_file = get_swarm_file_by_time(start_time, sat)
    end_time = tec_time + interval_radius + np.timedelta64(1, 'h')
    end_file = get_swarm_file_by_time(end_time, sat)
    timestamp, start_data = get_data_from_file(start_file, *fields)
    if end_file != start_file:
        data = []
        timestamp2, end_data = get_data_from_file(end_file, *fields)
        for s, e in zip(start_data, end_data):
            data.append(np.concatenate((s, e)))
        timestamp = np.concatenate((timestamp, timestamp2))
    else:
        data = start_data
    sl = slice(np.argmax(timestamp >= start_time), np.argmax(timestamp > end_time))
    return timestamp[sl], [d[sl] for d in data]


def get_closest_intervals_data(timestamp, mlat, tec_time):
    front = get_closest_segment(timestamp, mlat, tec_time, 45, 75)
    back = get_closest_segment(timestamp, mlat, tec_time, 75, 45)
    cap = get_closest_segment(timestamp, mlat, tec_time, 75)
    front_bot = get_closest_segment(timestamp, mlat, tec_time, 30, 45)
    back_bot = get_closest_segment(timestamp, mlat, tec_time, 45, 30)
    slices = sorted([front, back, cap, front_bot, back_bot], key=lambda x: x.start)
    lengths = [s.stop - s.start for s in slices]
    original_indexing = np.arange(timestamp.shape[0], dtype=int)
    new_indexing = np.empty(sum(lengths), dtype=int)
    acc = 0
    for i in range(5):
        new_indexing[acc:acc + lengths[i]] = original_indexing[slices[i]]
        acc += lengths[i]
    return front, back, new_indexing


def get_pixel_mask(mlt, mlat, radius):
    bins = [np.arange(29.5, 90), np.arange(-12 / 360, 24, 24 / 360)]
    path = binned_statistic_2d(mlat, mlt, np.ones(mlat.shape[0]), 'sum', bins=bins).statistic
    kernel = np.ones(radius + 1)[None, :]
    m = convolve2d(path, kernel, mode='same', boundary='wrap')
    return m > 0


def get_weight(mlt, mlat, bw=5, height=9):
    bins = [np.arange(29.5, 90), np.arange(-12 / 360, 24, 24 / 360)]
    path = binned_statistic_2d(mlat, mlt, np.ones(mlat.shape[0]), 'sum', bins=bins).statistic
    gamma = np.log(2) / bw ** 2
    width = 2 * np.ceil(bw * np.sqrt(-np.log(.01) / np.log(2))) + 1  # this far away, the exp will reach .01
    kernel = np.arange(-(width // 2), 1 + width // 2)[None, :] * np.ones(height)[:, None]
    kernel = np.exp(- gamma * kernel ** 2)
    w = convolve2d(path, kernel, mode='same', boundary='wrap')
    return w / w.max()


class SwarmDataInterval:
    """Holds an interval of swarm data. Accessing attributes will reach into the SWARM data dictionary

    plots:
        - closest segments within interval
    """

    def __init__(self, data, sl, tec_time=None):
        self.tec_time = tec_time
        self.data = {}
        for k, v in data.items():
            self.data[k] = v[sl]

    @classmethod
    def create_and_process(cls, year, month, index, sat, interval_radius=np.timedelta64(2, 'h')):
        day = index // 24 + 1
        hour = index % 24
        tec_time = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00")
        start_time = tec_time - interval_radius
        start_file = get_swarm_file_by_time(start_time, sat)
        if start_file is None:
            return None
        end_time = tec_time + interval_radius + np.timedelta64(1, 'h')
        end_file = get_swarm_file_by_time(end_time, sat)
        if end_file is None:
            return None
        start_data = cls.open_file(start_file)
        if np.mean(start_data['MLT'] == 0) > .5:
            return None
        if end_file != start_file:
            data = {}
            end_data = cls.open_file(end_file)
            if np.mean(end_data['MLT'] == 0) > .5:
                return None
            for f in start_data:
                data[f] = np.concatenate((start_data[f], end_data[f]))
        else:
            data = start_data
        sl = slice(np.argmax(data['Timestamp'] >= start_time), np.argmax(data['Timestamp'] > end_time))
        if sl.stop - sl.start < (2 * interval_radius.astype(int) + 1) * 60 * 60:
            return None
        interval = cls(data, sl, tec_time)
        interval.process()
        return interval

    @staticmethod
    def open_file(file):
        with pysatCDF.CDF(file) as f:
            data = f.data
        return data

    def process(self, median_window=3, mean_window=481):
        self.data['MLT'], self.data['MLat'] = fix_mlt_mlat(self.data['MLT'], self.data['MLat'])
        self.data['n'][self.data['n'] <= 0] = np.nan  # get rid of zeros
        logn = np.log10(self.data['n'])  # take log
        logn = centered_bn_func(bn.move_median, logn, median_window, min_count=1)  # median
        self.data = moving_func_trim(median_window, self.data)  # trim
        self.data['logn'] = logn
        background = centered_bn_func(bn.move_mean, self.data['logn'], mean_window, min_count=10)  # moving average
        self.data = moving_func_trim(mean_window, self.data)  # trim
        self.data['background'] = background

    def create_segment(self, index, find_troughs=True):
        return SwarmSegment(self, index, find_troughs)

    def get_closest_segments(self):
        front, back, total = get_closest_intervals_data(self.data['Timestamp'], self.data['MLat'], self.tec_time)
        return self.create_segment(front), self.create_segment(back), self.create_segment(total, False)


class SwarmSegment:
    """Holds and processes SWARM data for one "orbital segment"

    plots:
        - trough detection
        - tec trough intersection
    """

    def __init__(self, interval, index, find_troughs):
        self.data = {}
        for k, v in interval.data.items():
            self.data[k] = v[index]
        self.length = self.data['Timestamp'].shape[0]
        self.trough = False
        self.pwall_ind = None
        self.pwall_lat = None
        self.ewall_ind = None
        self.ewall_lat = None
        self.min_ind = None
        self.min_lat = None
        self.min_dne = None
        self.path_mask = None
        if find_troughs:
            self.find_trough()

    def get_tec_trough_intersection(self, tec_trough, radius=4):
        self.path_mask = get_pixel_mask(self.data['MLT'], self.data['MLat'], radius)
        return self.path_mask * tec_trough

    def find_trough(self, smooth_width=20):
        mlat = self.data['MLat']
        self.data['dne'] = self.data['logn'] - self.data['background']
        fin = np.isfinite(self.data['dne'])
        if np.any(~fin):
            self.data['dne'] = np.interp(np.arange(self.data['dne'].shape[0]),
                                         np.arange(self.data['dne'].shape[0])[fin], self.data['dne'][fin])
        self.data['smooth_dne'] = centered_bn_func(bn.move_mean, self.data['dne'], smooth_width, pad=True, min_count=1)
        trough = find_troughs_in_segment(mlat, self.data['smooth_dne'])
        if trough:
            self.trough = True
            self.min_ind, z1, z2 = trough
            self.min_lat = mlat[self.min_ind]
            self.min_dne = self.data['smooth_dne'][self.min_ind]
            if mlat[z1] > mlat[z2]:
                self.pwall_lat = mlat[z1]
                self.pwall_ind = z1
                self.ewall_lat = mlat[z2]
                self.ewall_ind = z2
            else:
                self.ewall_lat = mlat[z1]
                self.ewall_ind = z1
                self.pwall_lat = mlat[z2]
                self.pwall_ind = z2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    year = np.random.randint(2014, 2020)
    month = np.random.randint(1, 13)
    index = np.random.randint(0, 500)
    di = SwarmDataInterval.get_swarm_data_interval(year, month, index, 'A')
    print(di.Timestamp[0:5])
    di.Timestamp[0:5] = 0
    print(di.Timestamp[0:5])
