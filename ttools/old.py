ALL_SWARM_FIELDS = (
    'Latitude', 'Longitude', 'Height', 'Radius', 'SZA', 'SAz', 'ST', 'Diplat', 'Diplon', 'MLat', 'MLT', 'AACGMLat',
    'AACGMLon', 'n', 'Te_hgn', 'Te_lgn', 'T_elec', 'Vs_hgn', 'Vs_lgn', 'U_SC', 'Flagbits'
)


SWARM_FIELDS_LESS_MAG_COORDS = (
    'Height', 'Radius', 'SZA', 'SAz', 'ST', 'n', 'Te_hgn', 'Te_lgn', 'T_elec', 'Vs_hgn', 'Vs_lgn', 'U_SC', 'Flagbits'
)


SWARM_NEW_COORDS = (
    'apex_lat', 'apex_lon', 'qd_lat', 'qd_lon', 'mlt', 'lat', 'lon'
)

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

def get_s_data(start_date, end_date, sat, data_dir=None, coords_dir=None):
    """Gets SWARM data and timestamps assuming regular sampling. Fills in missing time steps with NaNs.

    Parameters
    ----------
    start_date, end_date: numpy.datetime64
    sat, data_dir, coords_dir: str

    Returns
    -------
    data: dict
    ref_times: numpy.ndarray[datetime64]
    """
    if data_dir is None:
        data_dir = config.swarm_dir
    if coords_dir is None:
        coords_dir = config.swarm_coords_dir
    fields = SWARM_FIELDS_LESS_MAG_COORDS + SWARM_NEW_COORDS

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