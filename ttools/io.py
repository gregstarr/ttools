import numpy as np
import datetime
import pandas
import os
import glob
import h5py
import yaml
from scipy import interpolate

from ttools import utils, config
from ttools.swarm import SWARM_SATELLITES


OMNI_COLUMNS = (
    "rotation_number", "imf_id", "sw_id", "imf_n", "plasma_n", "b_mag", "b_vector_mag", "b_vector_lat_avg",
    "b_vector_lon_avg", "bx", "by_gse", "bz_gse", "by_gsm", "bz_gsm", "b_mag_std", "b_vector_mag_std", "bx_std",
    "by_std", "bz_std", "proton_temp", "proton_density", "plasma_speed", "plasma_lon_angle", "plasma_lat_angle",
    "na_np_ratio", "flow_pressure", "temp_std", "density_std", "speed_std", "phi_v_std", "theta_v_std",
    "na_np_ratio_std", "e_field", "plasma_beta", "alfven_mach_number", "kp", "r", "dst", "ae", "proton_flux_1",
    "proton_flux_2", "proton_flux_4", "proton_flux_10", "proton_flux_30", "proton_flux_60", "proton_flux_flag", "ap",
    "f107", "pcn", "al", "au", "magnetosonic_mach_number"
)


def get_gm_index_kyoto(fn=None):
    if fn is None:
        fn = config.kp_file
    with open(fn, 'r') as f:
        text = f.readlines()
    ut_list = []
    kp_list = []
    ap_list = []
    for line in text[1:]:
        day = datetime.datetime.strptime(line[:8], '%Y%m%d')
        dt = datetime.timedelta(hours=3)
        uts = np.array([(day + i * dt).timestamp() for i in range(8)], dtype=int)
        kp = []
        for i in range(9, 25, 2):
            num = float(line[i])
            sign = line[i + 1]
            if sign == '+':
                num += 1 / 3
            elif sign == '-':
                num -= 1 / 3
            kp.append(num)
        kp_sum = float(line[25:27])
        sign = line[27]
        if sign == '+':
            kp_sum += 1 / 3
        elif sign == '-':
            kp_sum -= 1 / 3
        assert abs(kp_sum - sum(kp)) < .01
        kp_list.append(kp)
        ap = []
        for i in range(28, 52, 3):
            ap.append(float(line[i:i + 3]))
        ap = np.array(ap, dtype=int)
        Ap = float(line[52:55])
        ut_list.append(uts)
        ap_list.append(ap)

    ut = np.concatenate(ut_list)
    ap = np.concatenate(ap_list)
    kp = np.concatenate(kp_list)
    return pandas.DataFrame({'kp': kp, 'ap': ap, 'ut': ut}, index=pandas.to_datetime(ut, unit='s'))


def get_kp(times, fn=None):
    if fn is None:
        fn = config.kp_file
    data = get_gm_index_kyoto(fn)
    interpolator = interpolate.interp1d(data['ut'].values, data['kp'], kind='previous')
    return interpolator(times.astype('datetime64[s]').astype(float))


def get_omni_data(fn=None):
    if fn is None:
        fn = config.omni_file
    data = np.loadtxt(fn)
    year = (data[:, 0] - 1970).astype('datetime64[Y]')
    doy = (data[:, 1] - 1).astype('timedelta64[D]')
    hour = data[:, 2].astype('timedelta64[h]')
    datetimes = (year + doy + hour).astype('datetime64[s]')
    dtindex = pandas.DatetimeIndex(datetimes)
    df = pandas.DataFrame(data=data[:, 3:], index=dtindex, columns=OMNI_COLUMNS)
    for field in df:
        bad_val = df[field].max()
        bad_val_str = str(int(np.floor(bad_val)))
        if bad_val_str.count('9') == len(bad_val_str):
            mask = df[field] == bad_val
            df[field].loc[mask] = np.nan
    return df


def get_borovsky_data(fn="E:\\borovsky_2020_data.txt"):
    data = np.loadtxt(fn, skiprows=1)
    year = (data[:, 1] - 1970).astype('datetime64[Y]')
    doy = (data[:, 2] - 1).astype('timedelta64[D]')
    hour = data[:, 3].astype('timedelta64[h]')
    datetimes = (year + doy + hour).astype('datetime64[s]')
    return datetimes.astype(int), data[:, 4:]


def get_madrigal_data(start_date, end_date, data_dir=None):
    """Gets madrigal TEC and timestamps assuming regular sampling. Fills in missing time steps.

    Parameters
    ----------
    start_date, end_date: np.datetime64
    data_dir: str

    Returns
    -------
    tec, times: numpy.ndarray
    """
    if data_dir is None:
        data_dir = config.madrigal_dir
    dt = np.timedelta64(5, 'm')
    dt_sec = dt.astype('timedelta64[s]').astype(int)
    start_date = (np.ceil(start_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    end_date = (np.ceil(end_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    ref_times = np.arange(start_date, end_date, dt)
    ref_times_ut = ref_times.astype('datetime64[s]').astype(int)
    tec = np.ones((config.madrigal_lat.shape[0], config.madrigal_lon.shape[0], ref_times_ut.shape[0])) * np.nan
    file_dates = np.unique(ref_times.astype('datetime64[D]'))
    file_dates = utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        d = file_dates[i, 2]
        try:
            fn = glob.glob(os.path.join(data_dir, f"gps{y - 2000:02d}{m:02d}{d:02d}g.*.hdf5"))[-1]
        except IndexError:
            print(f"{y}-{m}-{d} madrigal file doesn't exist")
            continue
        t, ut, lat, lon = open_madrigal_file(fn)
        month_time_mask = np.in1d(ref_times_ut, ut)
        day_time_mask = np.in1d(ut, ref_times_ut)
        if not (np.all(lat == config.madrigal_lat) and np.all(lon == config.madrigal_lon)):
            print(f"THIS FILE HAS MISSING DATA!!!!!!! {fn}")
            lat_ind = np.argwhere(np.in1d(config.madrigal_lat, lat))[:, 0]
            lon_ind = np.argwhere(np.in1d(config.madrigal_lon, lon))[:, 0]
            time_ind = np.argwhere(month_time_mask)[:, 0]
            lat_grid_ind, lon_grid_ind, time_grid_ind = np.meshgrid(lat_ind, lon_ind, time_ind)
            tec[lat_grid_ind.ravel(), lon_grid_ind.ravel(), time_grid_ind.ravel()] = t[:, :, day_time_mask].ravel()
        else:
            # assume ut is increasing and has no repeating entries, basically that it is a subset of ref_times_ut
            tec[:, :, month_time_mask] = t[:, :, day_time_mask]
    return np.moveaxis(tec, -1, 0), ref_times


def open_madrigal_file(fn):
    """Open a madrigal file, return its data

    Parameters
    ----------
    fn: str
        madrigal file name to open

    Returns
    -------
    tec, timestamps, latitude, longitude: numpy.ndarray[float]
        (X, Y, T), (T, ), (X, ), (Y, )
    """
    with h5py.File(fn, 'r') as f:
        tec = f['Data']['Array Layout']['2D Parameters']['tec'][()]
        dtec = f['Data']['Array Layout']['2D Parameters']['tec'][()]
        timestamps = f['Data']['Array Layout']['timestamps'][()]
        lat = f['Data']['Array Layout']['gdlat'][()]
        lon = f['Data']['Array Layout']['glon'][()]
    print(f"Opened madrigal file: {fn}, size: {tec.shape}")
    return tec, timestamps, lat, lon


def get_swarm_data(start_date, end_date, data_dir=None):
    """Gets swarm and timestamps

    Parameters
    ----------
    start_date, end_date: np.datetime64
    data_dir: str

    Returns
    -------
    tec, times: numpy.ndarray
    """
    if data_dir is None:
        data_dir = config.swarm_dir
    dt = np.timedelta64(500, 'ms')
    dt_sec = dt.astype('timedelta64[ms]').astype(float)
    start_date = (np.ceil(start_date.astype('datetime64[ms]').astype(float) / dt_sec) * dt_sec).astype('datetime64[ms]')
    end_date = (np.ceil(end_date.astype('datetime64[ms]').astype(float) / dt_sec) * dt_sec).astype('datetime64[ms]')
    ref_times = np.arange(start_date, end_date, dt)
    ref_times_ut = ref_times.astype('datetime64[ms]').astype(float)
    keys = ['n', 'mlat', 'mlon', 'mlt']
    data = {sat: {key: [] for key in keys} for sat in SWARM_SATELLITES}
    file_dates = np.unique(ref_times.astype('datetime64[M]'))
    file_dates = utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        fn = os.path.join(data_dir, "{year:04d}_{month:02d}_swarm.h5".format(year=y, month=m))
        d, ut = open_swarm_file(fn)
        in_time_mask = np.in1d(ut, ref_times_ut)
        for sat in SWARM_SATELLITES:
            for key in keys:
                data[sat][key].append(d[sat][key][in_time_mask])
    for sat in SWARM_SATELLITES:
        for key in keys:
            data[sat][key] = np.concatenate(data[sat][key], axis=0)
    return data, ref_times


def open_swarm_file(fn):
    """Open a monthly SWARM file, return its data

    Parameters
    ----------
    fn: str

    Returns
    -------
    n, times, mlat, mlt, mlon: numpy.ndarray
    """
    data = {}
    with h5py.File(fn, 'r') as f:
        ut = f['ut_ms'][()]
        for sat in SWARM_SATELLITES:
            data[sat] = {
                'n': f[f'/swarm{sat}/n'][()],
                'mlat': f[f'/swarm{sat}/apex_lat'][()],
                'mlon': f[f'/swarm{sat}/apex_lon'][()],
                'mlt': f[f'/swarm{sat}/mlt'][()],
            }
    print(f"Opened SWARM file: {fn}, size: {ut.shape}")
    return data, ut


def get_tec_data(start_date, end_date, dt=np.timedelta64(1, 'h'), data_dir=None):
    """Gets TEC and timestamps

    Parameters
    ----------
    start_date, end_date: np.datetime64
    data_dir: str

    Returns
    -------
    tec, times: numpy.ndarray
    """
    if data_dir is None:
        data_dir = config.tec_dir
    dt_sec = dt.astype('timedelta64[s]').astype(int)
    start_date = (np.ceil(start_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    end_date = (np.ceil(end_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    ref_times = np.arange(start_date, end_date, dt)
    ref_times_ut = ref_times.astype('datetime64[s]').astype(int)
    tec = []
    ssmlon = []
    n_samples = []
    file_dates = np.unique(ref_times.astype('datetime64[M]'))
    file_dates = utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        fn = os.path.join(data_dir, "{year:04d}_{month:02d}_tec.h5".format(year=y, month=m))
        t, ut, ss, n, std = open_tec_file(fn)
        in_time_mask = np.in1d(ut, ref_times_ut)
        tec.append(t[in_time_mask])
        ssmlon.append(ss[in_time_mask])
        n_samples.append(n[in_time_mask])
    return np.concatenate(tec, axis=0), ref_times, np.concatenate(ssmlon), np.concatenate(n_samples)


def open_tec_file(fn):
    """Open a monthly TEC file, return its data

    Parameters
    ----------
    fn: str

    Returns
    -------
    tec, times, ssmlon, n, std: numpy.ndarray
    """
    with h5py.File(fn, 'r') as f:
        tec = f['tec'][()]
        n = f['n'][()]
        times = f['times'][()]
        std = f['std'][()]
        ssmlon = f['ssmlon'][()]
    print(f"Opened TEC file: {fn}, size: {tec.shape}")
    return tec, times, ssmlon, n, std


def get_arb_data(start_date, end_date, dt=np.timedelta64(1, 'h'), data_dir=None):
    """Gets auroral boundary mlat and timestamps

    Parameters
    ----------
    start_date, end_date: np.datetime64
    data_dir: str

    Returns
    -------
    arb_mlat, times: numpy.ndarray
    """
    if data_dir is None:
        data_dir = config.arb_dir
    dt_sec = dt.astype('timedelta64[s]').astype(int)
    start_date = (np.ceil(start_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    end_date = (np.ceil(end_date.astype('datetime64[s]').astype(int) / dt_sec) * dt_sec).astype('datetime64[s]')
    ref_times = np.arange(start_date, end_date, dt)
    ref_times_ut = ref_times.astype('datetime64[s]').astype(int)
    arb_mlat = []
    uts = []
    file_dates = np.unique(ref_times.astype('datetime64[M]'))
    file_dates = utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        fn = os.path.join(data_dir, f"{y:04d}_{m:02d}_arb.h5")
        mlat, ut = open_arb_file(fn)
        arb_mlat.append(mlat)
        uts.append(ut)
    uts = np.concatenate(uts)
    arb_mlat = np.concatenate(arb_mlat, axis=0)
    int_arb_mlat = np.empty((ref_times.shape[0], config.mlt_vals.shape[0]))
    for i in range(config.mlt_vals.shape[0]):
        int_arb_mlat[:, i] = np.interp(ref_times_ut, uts, arb_mlat[:, i])
    return int_arb_mlat, ref_times


def open_arb_file(fn):
    """Open a monthly auroral boundary file, return its data

    Parameters
    ----------
    fn: str

    Returns
    -------
    arb_mlat, times: numpy.ndarray
    """
    with h5py.File(fn, 'r') as f:
        arb_mlat = f['mlat'][()]
        times = f['times'][()]
    print(f"Opened ARB file: {fn}, size: {arb_mlat.shape}")
    return arb_mlat, times


def write_h5(fn, **kwargs):
    """Writes an h5 file with data specified by kwargs.

    Parameters
    ----------
    fn: str
        file path to write
    **kwargs
    """
    with h5py.File(fn, 'w') as f:
        for key, value in kwargs.items():
            f.create_dataset(key, data=value)


def write_yaml(fn, **kwargs):
    with open(fn, 'w') as f:
        yaml.safe_dump(kwargs, f)
