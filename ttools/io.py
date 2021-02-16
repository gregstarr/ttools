import numpy as np
import datetime
import pandas
import os
import glob
import h5py
import pysatCDF
import itertools
import yaml

from ttools import utils, config


OMNI_COLUMNS = (
    "rotation_number",
    "imf_id",
    "sw_id",
    "imf_n",
    "plasma_n",
    "b_mag",
    "b_vector_mag",
    "b_vector_lat_avg",
    "b_vector_lon_avg",
    "bx",
    "by_gse",
    "bz_gse",
    "by_gsm",
    "bz_gsm",
    "b_mag_std",
    "b_vector_mag_std",
    "bx_std",
    "by_std",
    "bz_std",
    "proton_temp",
    "proton_density",
    "plasma_speed",
    "plasma_lon_angle",
    "plasma_lat_angle",
    "na_np_ratio",
    "flow_pressure",
    "temp_std",
    "density_std",
    "speed_std",
    "phi_v_std",
    "theta_v_std",
    "na_np_ratio_std",
    "e_field",
    "plasma_beta",
    "alfven_mach_number",
    "kp",
    "r",
    "dst",
    "ae",
    "proton_flux_1",
    "proton_flux_2",
    "proton_flux_4",
    "proton_flux_10",
    "proton_flux_30",
    "proton_flux_60",
    "proton_flux_flag",
    "ap",
    "f107",
    "pcn",
    "al",
    "au",
    "magnetosonic_mach_number",
)

ALL_SWARM_FIELDS = (
    'Latitude',
    'Longitude',
    'Height',
    'Radius',
    'SZA',
    'SAz',
    'ST',
    'Diplat',
    'Diplon',
    'MLat',
    'MLT',
    'AACGMLat',
    'AACGMLon',
    'n',
    'Te_hgn',
    'Te_lgn',
    'T_elec',
    'Vs_hgn',
    'Vs_lgn',
    'U_SC',
    'Flagbits',
)


SWARM_FIELDS_LESS_MAG_COORDS = (
    'Height',
    'Radius',
    'SZA',
    'SAz',
    'ST',
    'n',
    'Te_hgn',
    'Te_lgn',
    'T_elec',
    'Vs_hgn',
    'Vs_lgn',
    'U_SC',
    'Flagbits',
)


SWARM_NEW_COORDS = (
    'apex_lat',
    'apex_lon',
    'qd_lat',
    'qd_lon',
    'mlt',
    'lat',
    'lon',
)


def get_gm_index_kyoto(fn="E:\\2000_2020_kp_ap.txt"):
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


def get_omni_data(fn="E:\\omni2_all_years.dat"):
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


def get_madrigal_data(start_date, end_date, dir=config.madrigal_dir):
    """Gets madrigal TEC and timestamps assuming regular sampling. Fills in missing time steps.

    Parameters
    ----------
    start_date, end_date: np.datetime64
    dir: str

    Returns
    -------
    tec, times: numpy.ndarray
    """
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
            fn = glob.glob(os.path.join(dir, f"gps{y - 2000:02d}{m:02d}{d:02d}g.*.hdf5"))[-1]
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


def get_swarm_data(start_date, end_date, sat, data_dir=config.swarm_dir, coords_dir=config.swarm_coords_dir):
    """Gets madrigal TEC and timestamps assuming regular sampling. Fills in missing time steps with NaNs.

    Parameters
    ----------
    start_date, end_date: numpy.datetime64
    sat, data_dir, coords_dir: str

    Returns
    -------
    data: dict
    ref_times: numpy.ndarray[datetime64]
    """
    if coords_dir is not None:
        fields = SWARM_FIELDS_LESS_MAG_COORDS + SWARM_NEW_COORDS
    else:
        fields = ALL_SWARM_FIELDS

    dt = np.timedelta64(500, 'ms')
    dt_sec = dt.astype('timedelta64[ms]').astype(float)
    start_date = (np.ceil(start_date.astype('datetime64[ms]').astype(float) / dt_sec) * dt_sec).astype('datetime64[ms]')
    end_date = (np.ceil(end_date.astype('datetime64[ms]').astype(float) / dt_sec) * dt_sec).astype('datetime64[ms]')
    ref_times = np.arange(start_date, end_date, dt)
    ref_times_ut = ref_times.astype('datetime64[ms]').astype(float)
    data = {f: np.ones(ref_times.shape[0]) * np.nan for f in fields}
    file_dates = np.unique(ref_times.astype('datetime64[D]'))
    file_dates = utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        d = file_dates[i, 2]
        files = glob.glob(os.path.join(data_dir, f"SW_EXTD_EFI{sat.upper()}_LP_HM_{y:04d}{m:02d}{d:02d}*.cdf"))
        files = filter_swarm_files(files)
        for fn in files:
            file_data = open_swarm_file(fn)
            if coords_dir is not None:
                coords_fn = os.path.join(coords_dir, f"{utils.no_ext_fn(fn)}_coords.h5")
                file_data.update(open_swarm_coords_file(coords_fn))
            file_times_ut = (np.floor(file_data['Timestamp'].astype('datetime64[ms]').astype(float) / dt_sec) * dt_sec)
            # assume ut is increasing and has no repeating entries, basically that it is a subset of ref_times_ut
            r_mask = np.in1d(ref_times_ut, file_times_ut)
            c_mask = np.in1d(file_times_ut, ref_times_ut)
            for f in fields:
                if f in file_data:
                    data[f][r_mask] = file_data[f][c_mask]
    return data, ref_times


def filter_swarm_files(files):
    """given a list of SWARM filenames, returns a list only including the latest version of each file

    Parameters
    ----------
    files: list[str]

    Returns
    -------
    list[str]
    """
    result = []
    base = [(os.path.split(fn)[0], utils.no_ext_fn(fn)) for fn in files]
    splitup = [(b[:-4], b[-4:], a) for a, b in base]
    splitup = sorted(splitup, key=lambda x: x[0])
    for key, grp in itertools.groupby(splitup, lambda x: x[0]):
        latest_version = sorted(grp, key=lambda x: x[1])[-1]
        result.append(os.path.join(latest_version[2], f"{latest_version[0]}{latest_version[1]}.cdf"))
    return result


def open_swarm_file(fn):
    """Opens a SWARM file

    Parameters
    ----------
    fn: str

    Returns
    -------
    dict
        Timestamp
        Latitude
        Longitude
        Height: m Height above WGS84 reference ellipsoid.
        Radius: m Distance from the Earth’s centre.
        SZA: deg Solar Zenith Angle.
        SAz: deg Solar azimuth in Earth frame, north is 0 deg.
        ST: hour Apparent solar time
        Diplat: deg Quasi-dipole latitude
        Diplon
        MLT: hour Magnetic local time based on quasi-dipole
        AACGMLat: deg Altitude-adjusted corrected geomagnetic latitude
        AACGMLon
        n: cm-3 Plasma density from ion current
        Te_hgn: K Electron temperature, estimated by the high gain probe
        Te_lgn: K Electron temperature, estimated by the low gain probe
        Te: K Electron temperature, blended value
        Vs_hgn: V Spacecraft potential, estimated by the high gain probe
        Vs_lgn: V Spacecraft potential, estimated by the low gain probe
        Vs: V Spacecraft potential, blended value
        Flagbits
    """
    with pysatCDF.CDF(fn) as f:
        data = f.data
    print(f"Opened swarm file: {fn}, size: {data['Timestamp'].shape}")
    return data


def open_swarm_coords_file(fn):
    """Opens precomputed swarm coordinates h5 file (one per swarm cdf file)

    Parameters
    ----------
    fn: str

    Returns
    -------
    dict of numpy.ndarray[float]
        keys: 'apex_lat', 'apex_lon', 'qd_lat', 'qd_lon', 'mlt'
    """
    coords = {}
    with h5py.File(fn, 'r') as f:
        coords['apex_lat'] = f['apex_lat'][()]
        coords['apex_lon'] = f['apex_lon'][()]
        coords['qd_lat'] = f['qd_lat'][()]
        coords['qd_lon'] = f['qd_lon'][()]
        coords['mlt'] = f['mlt'][()]
        coords['lat'] = f['lat'][()]
        coords['lon'] = f['lon'][()]
    print(f"Opened swarm coords file: {fn}, size: {coords['apex_lat'].shape}")
    return coords


def get_tec_data(start_date, end_date, dir=config.tec_dir):
    """Gets TEC and timestamps

    Parameters
    ----------
    start_date, end_date: np.datetime64
    dir: str

    Returns
    -------
    tec, times: numpy.ndarray
    """
    dt = np.timedelta64(1, 'h')
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
        fn = os.path.join(dir, "{year:04d}_{month:02d}_tec.h5".format(year=y, month=m))
        print(fn)
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
