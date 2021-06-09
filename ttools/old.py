import os, glob, numpy as np, pysatCDF
from ttools import config, utils


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
    import os, itertools
    from ttools import utils
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
    import h5py
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


def get_swarm_data(start_date, end_date, sat, data_dir=None, coords_dir=None):
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

    dt = np.timedelta64(500, 'ms')
    dt_sec = dt.astype('timedelta64[ms]').astype(float)
    start_date = (np.ceil(start_date.astype('datetime64[ms]').astype(float) / dt_sec) * dt_sec).astype('datetime64[ms]')
    end_date = (np.ceil(end_date.astype('datetime64[ms]').astype(float) / dt_sec) * dt_sec).astype('datetime64[ms]')
    ref_times = np.arange(start_date, end_date, dt)
    ref_times_ut = ref_times.astype('datetime64[ms]').astype(float)
    data = {}
    file_dates = np.unique(ref_times.astype('datetime64[D]'))
    file_dates = utils.decompose_datetime64(file_dates)
    for i in range(file_dates.shape[0]):
        y = file_dates[i, 0]
        m = file_dates[i, 1]
        d = file_dates[i, 2]
        lp_files = glob.glob(os.path.join(data_dir, 'extd_efi_lp', f"SW_EXTD_EFI{sat.upper()}_LP_HM_{y:04d}{m:02d}{d:02d}*.cdf"))
        lp_files = filter_swarm_files(lp_files)
        efi_files = glob.glob(os.path.join(data_dir, 'expt_efi_tct02', f"SW_EXPT_EFI{sat.upper()}_TCT02_{y:04d}{m:02d}{d:02d}*.cdf"))
        efi_files = filter_swarm_files(efi_files)
        for fn in efi_files + lp_files:
            file_data = open_swarm_file(fn)
            coords_fn = os.path.join(coords_dir, f"{utils.no_ext_fn(fn)}_coords.h5")
            if os.path.isfile(coords_fn):
                file_data.update(open_swarm_coords_file(coords_fn))
            file_times_ut = (np.floor(file_data['Timestamp'].astype('datetime64[ms]').astype(float) / dt_sec) * dt_sec)
            # assume ut is increasing and has no repeating entries, basically that it is a subset of ref_times_ut
            r_mask = np.in1d(ref_times_ut, file_times_ut)
            c_mask = np.in1d(file_times_ut, ref_times_ut)
            for k, v in file_data.items():
                if k not in data:
                    data[k] = np.ones(ref_times.shape[0]) * np.nan
                data[k][r_mask] = v[c_mask]
    return data, ref_times


"""
Create trough dataset:
    0: No trough
    1: Non-SAPS trough
    2: SAPS trough
    3: unknown trough

Procedure:
    1. identify what trough connected component DMSP is passing trough, label the CC as saps or non-saps
    2. identify location of 

Create plots:
    (low, medium, high Kp) x (SAPS, no SAPS)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd
import bottleneck as bn

from ttools import io, plotting, utils, config, satellite

trough_data = np.load("C:\\Users\\Greg\\data\\dataset.npz")
trough = trough_data['trough']
tec_times = trough_data['time']
tec_ut = tec_times.astype(int)
X = -1 * trough_data['x']
X[~trough] = np.nan
kp = io.get_kp(tec_times)

if True:
    batch_size = 100
    flow_grid_count = np.empty_like(X)
    flow_grid_sum = np.empty_like(X)
    for batch in range(int(np.ceil(trough.shape[0] / batch_size))):
        print(batch, np.ceil(trough.shape[0] / batch_size))
        i1 = batch * batch_size
        i2 = min((batch + 1) * batch_size, trough.shape[0] - 1)
        start = tec_times[i1]
        end = tec_times[i2] + np.timedelta64(1, 'h')
        dmsp, dmsp_times = io.get_dmsp_data(start, end)

        bins = [np.arange(start, end, np.timedelta64(1, 'h')), np.arange(29.5, 90),
                np.arange(-12, 12 + 24 / 360, 48 / 360)]
        t = []
        mlat = []
        mlt = []
        flow = []
        for sat, sat_data in dmsp.items():
            t.append(dmsp_times.astype(int))
            mlat.append(sat_data['mlat'])
            mlt.append(sat_data['mlt'])
            flow.append(sat_data['hor_ion_v'])
        t, mlat, mlt, flow = utils.concatenate(t, mlat, mlt, flow)
        mlt[mlt > 12] -= 24
        sample = np.column_stack((t, mlat, mlt))
        mask = (mlat >= 30) & np.isfinite(flow)
        flow_grid_count[i1:i2] = binned_statistic_dd(sample[mask], flow[mask], 'count', bins).statistic
        flow_grid_sum[i1:i2] = binned_statistic_dd(sample[mask], flow[mask], 'sum', bins).statistic

        starts, stops = satellite.get_closest_segment(dmsp_times, dmsp['dmsp16']['mlat'], tec_times[i1:i2], 30)
        for start, stop in zip(starts, stops):
            saps = satellite.find_troughs_in_segment(dmsp['dmsp16']['mlat'][start:stop],
                                                     -dmsp['dmsp16']['hor_ion_v'][start:stop], -300)
            plt.plot(dmsp['dmsp16']['hor_ion_v'][start:stop])
            for sap in saps:
                m, e1, e2 = sap
                plt.plot(m, dmsp['dmsp16']['hor_ion_v'][start:stop][m], 'rx')
            plt.show()

    np.savez('E:\\dmsp_flow\\flow_grid.npz', flow_grid_count=flow_grid_count, flow_grid_sum=flow_grid_sum)

# fig, ax = plt.subplots(subplot_kw={'polar': True})
# for sat, sat_data in dmsp.items():
#     m = sat_data['mlat'] > 30
#     ax.plot((sat_data['mlt'][m] - 6) * np.pi / 12, 90 - sat_data['mlat'][m], '.', label=sat)
# plotting.format_polar_mag_ax(ax)
# plt.legend()
# plt.show()

flow_data = np.load('E:\\dmsp_flow\\flow_grid.npz')
fg_sum = flow_data['flow_grid_sum']
fg_count = flow_data['flow_grid_count']
fg_sum[~trough] = 0
fg_count[~trough] = 0

fg = fg_sum / fg_count
saps = fg > 500
non_saps = fg < 500
"""