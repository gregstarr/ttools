import numpy as np
from scipy import signal
import bottleneck as bn

from ttools import swarm, io, utils


def test_fix_latlon():
    """Verify `fix_latlon` fixes geodetic / geographic coordinate averaging errors (at 0/360 border)
    """
    lon = np.arange(200, 400)
    lon %= 360
    lat = np.ones_like(lon) * 10
    lon = bn.move_mean(lon, window=3, min_count=1)
    x = np.cos(np.radians(lon)) * (90 - lat)
    y = np.sin(np.radians(lon)) * (90 - lat)
    z = np.hypot(np.diff(x), np.diff(y))
    fixed_lat, fixed_lon = swarm.fix_latlon(lat, lon)
    xf = np.cos(np.radians(fixed_lon)) * (90 - fixed_lat)
    yf = np.sin(np.radians(fixed_lon)) * (90 - fixed_lat)
    zf = np.hypot(np.diff(xf), np.diff(yf))
    assert np.sum(z > 5) > 1
    assert np.sum(zf > 5) == 0


def test_process_swarm_data_interval():
    """Verify that the shape of the output arrays changes as expected
    """
    N = 100
    times = np.datetime64('2015-10-10T10:10:10') + np.arange(N) * np.timedelta64(500, 'ms')
    logne = np.random.randn(N)
    data = {'n': np.exp(logne), 'apex_lat': np.random.rand(N), 'mlt': np.random.rand(N)}
    new_t, new_ln, bg, mlat, mlt = swarm.process_swarm_data_interval(data, times, median_window=21, mean_window=21)
    assert new_t.shape[0] + 40 == times.shape[0]
    assert new_ln.shape[0] + 40 == logne.shape[0]


def test_get_enter_exit():
    N = 1000
    periods = 2.5
    mlat = signal.sawtooth(np.linspace(0, 2 * periods * np.pi, N), .5) * 5 + 5
    # nominal front
    enter_mask = mlat >= 3
    exit_mask = mlat > 7
    starts, ends = swarm.get_region_bounds(enter_mask, exit_mask)
    assert np.all(mlat[starts] >= 3)
    assert np.all(mlat[starts - 1] < 3)
    assert np.all(mlat[ends] > 7)
    assert np.all(mlat[ends - 1] <= 7)
    # nominal back
    enter_mask = mlat <= 7
    exit_mask = mlat < 3
    starts, ends = swarm.get_region_bounds(enter_mask, exit_mask)
    assert np.all(mlat[starts] <= 7)
    assert np.all(mlat[starts - 1] > 7)
    assert np.all(mlat[ends] < 3)
    assert np.all(mlat[ends - 1] >= 3)
    # cap
    enter_mask = mlat >= 7
    exit_mask = mlat < 7
    starts, ends = swarm.get_region_bounds(enter_mask, exit_mask)
    assert np.all(mlat[starts] >= 7)
    assert np.all(mlat[starts - 1] < 7)
    assert np.all(mlat[ends] < 7)
    assert np.all(mlat[ends - 1] >= 7)
    # anti cap
    enter_mask = mlat <= 3
    exit_mask = mlat > 3
    starts, ends = swarm.get_region_bounds(enter_mask, exit_mask)
    assert np.all(mlat[starts] <= 3)
    assert np.all(mlat[starts - 1] > 3)
    assert np.all(mlat[ends] > 3)
    assert np.all(mlat[ends - 1] <= 3)
    # no full region
    enter_mask = mlat[round(N * .25 / periods):round(N * 1.25 / periods)] >= 3
    exit_mask = mlat[round(N * .25 / periods):round(N * 1.25 / periods)] > 7
    starts, ends = swarm.get_region_bounds(enter_mask, exit_mask)
    assert starts.size == 0
    assert ends.size == 0


def test_get_enter_exit_data():
    start_date = np.datetime64("2015-10-07T00:00:00")
    end_date = np.datetime64("2015-10-08T00:00:00.000")
    data, times = io.get_swarm_data(start_date, end_date, 'C')
    mlat = data['apex_lat']
    fin_ind = np.argwhere(np.isfinite(mlat))[:, 0]
    enter_mask = mlat[fin_ind] >= 45
    exit_mask = mlat[fin_ind] >= 75
    starts, ends = swarm.get_region_bounds(enter_mask, exit_mask)
    assert np.all(mlat[fin_ind[starts]] >= 45)
    assert np.all(mlat[fin_ind[starts - 1]] < 45)
    assert np.all(mlat[fin_ind[ends]] > 75)
    assert np.all(mlat[fin_ind[ends - 1]] <= 75)


def test_get_closest_segment():
    start_date = np.datetime64("2015-10-07T06:00:00")
    end_date = np.datetime64("2015-10-07T12:00:00.000")
    data, times = io.get_swarm_data(start_date, end_date, 'C')
    fin_mask = np.isfinite(data['apex_lat'])
    tec_times = start_date + np.arange(6) * np.timedelta64(1, 'h')
    starts, stops = swarm.get_closest_segment(times[fin_mask], data['apex_lat'][fin_mask], tec_times, 45, 75)
    for start, stop in zip(starts, stops):
        assert np.all(data['apex_lat'][fin_mask][start:stop] >= 45)
        assert data['apex_lat'][fin_mask][start - 1] < 45
        assert np.all(data['apex_lat'][fin_mask][start:stop] < 75)
        assert data['apex_lat'][fin_mask][stop] >= 75

    starts, stops = swarm.get_closest_segment(times[fin_mask], data['apex_lat'][fin_mask], tec_times, 75, 45)
    for start, stop in zip(starts, stops):
        assert np.all(data['apex_lat'][fin_mask][start:stop] >= 45)
        assert data['apex_lat'][fin_mask][start - 1] >= 75
        assert np.all(data['apex_lat'][fin_mask][start:stop] < 75)
        assert data['apex_lat'][fin_mask][stop] < 45

    starts, stops = swarm.get_closest_segment(times[fin_mask], data['apex_lat'][fin_mask], tec_times, 75)
    for start, stop in zip(starts, stops):
        assert np.all(data['apex_lat'][fin_mask][start:stop] >= 75)
        assert data['apex_lat'][fin_mask][start - 1] < 75
        assert data['apex_lat'][fin_mask][stop] < 75


def test_find_troughs_in_segment_nominal():
    threshold = -1
    mlat = np.arange(100)
    smooth_dne = np.zeros(100)
    smooth_dne[25:50] = np.linspace(0, -2, 25)  # 25 is 0
    smooth_dne[50:75] = np.linspace(-2.1, -.1, 25)  # 75 is 0, min at 50
    trough = swarm.find_troughs_in_segment(mlat, smooth_dne, threshold, width_max=50)
    assert trough
    tmin, e1, e2 = trough
    assert e1 == 26
    assert e2 == 75
    assert tmin == 50


def test_find_troughs_in_segment_missing_1():
    """missing outside trough
    """
    threshold = -1
    mlat = np.arange(100)
    smooth_dne = np.zeros(100)
    smooth_dne[25:50] = np.linspace(0, -2, 25)  # 25 is 0
    smooth_dne[50:75] = np.linspace(-2.1, -.1, 25)  # 75 is 0, min at 50
    smooth_dne[10:20] = np.nan
    trough = swarm.find_troughs_in_segment(mlat, smooth_dne, threshold, width_max=50)
    assert trough
    tmin, e1, e2 = trough
    assert e1 == 26
    assert e2 == 75
    assert tmin == 50


def test_find_troughs_in_segment_missing_2():
    """missing inside trough
    """
    threshold = -1
    mlat = np.arange(100)
    smooth_dne = np.zeros(100)
    smooth_dne[25:50] = np.linspace(0, -2, 25)  # 25 is 0
    smooth_dne[50:75] = np.linspace(-2.1, -.1, 25)  # 75 is 0, min at 50
    smooth_dne[60:70] = np.nan
    trough = swarm.find_troughs_in_segment(mlat, smooth_dne, threshold, width_max=50)
    assert trough
    tmin, e1, e2 = trough
    assert e1 == 26
    assert e2 == 75
    assert tmin == 50


def test_find_troughs_in_segment_missing_3():
    """missing overlapping inside and outside trough
    """
    threshold = -1
    mlat = np.arange(100)
    smooth_dne = np.zeros(100)
    smooth_dne[25:50] = np.linspace(0, -2, 25)  # 25 is 0
    smooth_dne[50:75] = np.linspace(-2.1, -.1, 25)  # 75 is 0, min at 50
    smooth_dne[60:80] = np.nan
    trough = swarm.find_troughs_in_segment(mlat, smooth_dne, threshold, width_max=50)
    assert not trough
    trough = swarm.find_troughs_in_segment(mlat, smooth_dne, threshold, width_max=60)
    tmin, e1, e2 = trough
    assert e1 == 26
    assert e2 == 80
    assert tmin == 50


def test_find_troughs_in_segment_data():
    start_date = np.datetime64("2015-10-07T06:00:00")
    end_date = np.datetime64("2015-10-07T12:00:00.000")
    data, times = io.get_swarm_data(start_date, end_date, 'C')
    times, logne, background, mlat, mlt = swarm.process_swarm_data_interval(data, times)
    dne = logne - background
    smooth_dne = utils.centered_bn_func(bn.move_mean, dne, 11, pad=True, min_count=1)
    fin_mask = np.isfinite(mlat)

    tec_times = start_date + np.arange(6) * np.timedelta64(1, 'h')
    front_starts, front_stops = swarm.get_closest_segment(times[fin_mask], mlat[fin_mask], tec_times, 45, 75)
    back_starts, back_stops = swarm.get_closest_segment(times[fin_mask], mlat[fin_mask], tec_times, 75, 45)
    for fstart, fstop, bstart, bstop in zip(front_starts, front_stops, back_starts, back_stops):
        front = slice(fstart, fstop)
        back = slice(bstart, bstop)
        threshold = -.2
        front_trough = swarm.find_troughs_in_segment(mlat[fin_mask][front], smooth_dne[fin_mask][front], threshold)
        back_trough = swarm.find_troughs_in_segment(mlat[fin_mask][back], smooth_dne[fin_mask][back], threshold)
        for side, trough in zip([front, back], [front_trough, back_trough]):
            if trough:
                min_idx, edge1, edge2 = trough
                assert np.any(smooth_dne[fin_mask][side][min_idx] < threshold)
                assert edge1 <= min_idx <= edge2

        threshold = -10
        front_trough = swarm.find_troughs_in_segment(mlat[fin_mask][front], smooth_dne[fin_mask][front], threshold)
        back_trough = swarm.find_troughs_in_segment(mlat[fin_mask][back], smooth_dne[fin_mask][back], threshold)
        assert not (front_trough or back_trough)

        threshold = -.01
        front_trough = swarm.find_troughs_in_segment(mlat[fin_mask][front], smooth_dne[fin_mask][front], threshold)
        back_trough = swarm.find_troughs_in_segment(mlat[fin_mask][back], smooth_dne[fin_mask][back], threshold)
        assert front_trough and back_trough


def test_get_segments_data():
    T = 12
    one_h = np.timedelta64(1, 'h')
    start_time = np.datetime64("2015-10-07T06:00:00")
    end_time = start_time + np.timedelta64(T, 'h')
    tec_times = np.arange(start_time, end_time, one_h)
    segments = swarm.get_segments_data(tec_times)
    for sat, sat_segments in segments.items():
        for direction, dir_segments in sat_segments.items():
            for seg in dir_segments:
                assert np.all(abs(seg['times'] - seg['tec_time']).astype('timedelta64[s]').astype(float) < 60 * 60)
                if direction == 'up':
                    assert np.all(np.diff(seg['mlat']) > 0)
                elif direction == 'down':
                    assert np.all(np.diff(seg['mlat']) < 0)
                else:
                    assert False


def test_get_swarm_troughs():
    T = 12
    one_h = np.timedelta64(1, 'h')
    start_time = np.datetime64("2015-10-07T06:00:00")
    end_time = start_time + np.timedelta64(T, 'h')
    tec_times = np.arange(start_time, end_time, one_h)
    segments = swarm.get_segments_data(tec_times)
    troughs = swarm.get_swarm_troughs(segments)
    for _, t in troughs.iterrows():
        sat = t['sat']
        direction = 'up' if abs(t['seg_e1_mlat'] - 45) < 2 else 'down'
        i = t['tec_ind']
        print(_, np.any(segments[sat][direction][i]['smooth_dne'] < -.2), t['trough'])
        assert np.any(segments[sat][direction][i]['smooth_dne'] < -.2) == t['trough']
