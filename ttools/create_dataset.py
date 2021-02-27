import numpy as np
import apexpy
from scipy.stats import binned_statistic_2d
import time
from multiprocessing import Pool
import os

from ttools import io, config, convert, utils


def assemble_binning_args(mlat, mlt, tec, times, ssmlon, bins, map_period):
    """Creates a list of tuple arguments to be passed to `calculate_bins`. `calculate_bins` is called by the process
    pool manager using each tuple in the list returned by this function as arguments. Each set of arguments corresponds
    to one processed TEC map and should span a time period specified by `map_period`. `map_period` should evenly divide
    24h, and probably should be a multiple of 5 min.

    Parameters
    ----------
    mlat, mlt, tec, times, ssmlon: numpy.ndarray[float]
    bins: list[numpy.ndarray[float]]
    map_period: {np.timedelta64, int}

    Returns
    -------
    list[tuple]
        each tuple in the list is passed to `calculate_bins`
    """
    if isinstance(map_period, np.timedelta64):
        map_period = map_period.astype('timedelta64[s]').astype(int)
    args = []
    current_time = times[0]
    while current_time < times[-1]:
        start = np.argmax(times >= current_time)
        end = np.argmax(times >= current_time + map_period)
        if end == 0:
            end = times.shape[0]
        time_slice = slice(start, end)
        fin_mask = np.isfinite(tec[time_slice])
        mlat_r = mlat[time_slice][fin_mask].copy()
        mlt_r = mlt[time_slice][fin_mask].copy()
        tec_r = tec[time_slice][fin_mask].copy()
        args.append((mlat_r, mlt_r, tec_r, times[time_slice], ssmlon[time_slice], bins))
        current_time += map_period
    return args


def calculate_bins(mlat, mlt, tec, times, ssmlon, bins):
    """Calculates TEC in MLAT - MLT bins. Executed in process pool.

    Parameters
    ----------
    mlat, mlt, tec, times, ssmlon: numpy.ndarray[float] (N, )
    bins: list[numpy.ndarray[float] (X + 1, ), numpy.ndarray[float] (Y + 1, )]

    Returns
    -------
    tuple
        time: int or float
        final_tec, final_tec_n, final_tec_s: numpy.ndarray[float] (T, X, Y)
    """
    if tec.size == 0:
        placeholder = np.ones((bins[0].shape[0] - 1, bins[1].shape[0] - 1)) * np.nan
        final_tec = placeholder.copy()
        final_tec_n = placeholder.copy()
        final_tec_s = placeholder.copy()
    else:
        final_tec = binned_statistic_2d(mlat, mlt, tec, 'mean', bins).statistic
        final_tec_n = binned_statistic_2d(mlat, mlt, tec, 'count', bins).statistic
        final_tec_s = binned_statistic_2d(mlat, mlt, tec, 'std', bins).statistic
    return times[0], final_tec, ssmlon[0], final_tec_n, final_tec_s


def process_file(start_date, end_date, mlat_grid, mlon_grid, converter, bins, map_period=np.timedelta64(1, 'h'),
                 madrigal_dir=None):
    """Processes madrigal data into a single file. Opens madrigal h5 files, converts input mlon grid to MLT by
    computing subsolar points at each time step, sets up and runs TEC binning, unpacks and returns results.

    Parameters
    ----------
    start_date, end_date: np.datetime64
    mlat_grid, mlon_grid: numpy.ndarray[float] (X, Y)
    converter: apexpy.Apex
    bins: list[numpy.ndarray[float] (X + 1, ), numpy.ndarray[float] (Y + 1, )]
    map_period: {np.timedelta64, int}
    madrigal_dir: str
        to specify alternate directory during testing

    Returns
    -------
    times, tec, n, std, ssmlon: numpy.ndarray[float]
            (T, ), (T, X, Y), (T, X, Y), (T, X, Y), (T, )
    """
    if not isinstance(madrigal_dir, str):
        madrigal_dir = config.madrigal_dir
    print(start_date, end_date)
    tec, ts = io.get_madrigal_data(start_date, end_date, data_dir=madrigal_dir)
    print("Converting coordinates")
    mlt, ssmlon = convert.mlon_to_mlt_array(mlon_grid[None, :, :], ts[:, None, None], converter, return_ssmlon=True)
    mlat = mlat_grid[None, :, :] * np.ones((ts.shape[0], 1, 1))
    mlt[mlt > 12] -= 24
    print("Setting up for binning")
    args = assemble_binning_args(mlat, mlt, tec, ts, ssmlon, bins, map_period)
    print(f"Calculating bins for {len(args)} time steps")
    with Pool(processes=8) as p:
        pool_result = p.starmap(calculate_bins, args)
    print("Calculated bins")
    times = np.array([r[0] for r in pool_result])
    tec = np.array([r[1] for r in pool_result])
    ssmlon = np.array([r[2] for r in pool_result])
    n = np.array([r[3] for r in pool_result])
    std = np.array([r[4] for r in pool_result])
    return times, tec, ssmlon, n, std


def get_mag_grid(ref_lat, ref_lon, converter):
    lon_grid, lat_grid = np.meshgrid(ref_lon, ref_lat)
    mlat, mlon = converter.convert(lat_grid.ravel(), lon_grid.ravel(), 'geo', 'apex', height=350)
    mlat = mlat.reshape(lat_grid.shape)
    mlon = mlon.reshape(lat_grid.shape)
    return mlat, mlon


def _output_file_name(date):
    ymd = utils.decompose_datetime64(date)
    return "{year:04d}_{month:02d}_tec.h5".format(year=ymd[0, 0], month=ymd[0, 1])


def process_multiple_files(start_date, end_date, bins, dt=np.timedelta64(1, 'M'), map_period=np.timedelta64(1, 'h'),
                           ref_lat=None, ref_lon=None, output_dir=None, file_name_pattern=_output_file_name):
    """Processes an interval of madrigal data and writes to files.

    Parameters
    ----------
    start_date, end_date: numpy.datetime64
    bins: list[numpy.ndarray[float] (X + 1, ), numpy.ndarray[float] (Y + 1, )]
    dt, map_period: numpy.timedelta64
    ref_lat, ref_lon: numpy.ndarray[float]
        latitude / longitude values of madrigal lat-lon grid
    output_dir: str
        directory to write files to
    file_name_pattern: callable
    """
    if ref_lat is None:
        ref_lat = config.madrigal_lat
    if ref_lon is None:
        ref_lon = config.madrigal_lon
    if output_dir is None:
        output_dir = config.tec_dir
    apex_date = utils.datetime64_to_datetime(start_date)
    converter = apexpy.Apex(date=apex_date)
    mlat, mlon = get_mag_grid(ref_lat, ref_lon, converter)
    months = np.arange(start_date, end_date, dt)
    for month in months:
        times, tec, ssmlon, n, std = process_file(month, min(end_date, month + dt), mlat, mlon, converter, bins,
                                                  map_period)
        if np.isfinite(tec).any():
            fn = os.path.join(output_dir, file_name_pattern(month))
            io.write_h5(fn, times=times.astype('datetime64[s]').astype(int), tec=tec, n=n, std=std, ssmlon=ssmlon)
        else:
            print(f"No data for month: {month}, not writing file")


def process_dataset(start_year, end_year, mlat_bins, mlt_bins, apex_dt=np.timedelta64(1, 'Y'),
                    file_dt=np.timedelta64(1, 'M'), map_period=np.timedelta64(1, 'h'), output_dir=None,
                    file_name_pattern=_output_file_name):
    """'main' function for creating tec dataset. Brief setup, then calls `process_multiple_files` on each year as
    specified by inputs. Writes grid file.

    Parameters
    ----------
    start_year, end_year: numpy.datetime64
    mlat_bins, mlt_bins: numpy.ndarray[float]
    apex_dt, file_dt, map_period: numpy.timedelta
        interval to update apex date, interval to write to each file, interval to include in each map
    output_dir: str
    file_name_pattern: callable
        function accepts a numpy.datetime and returns an h5 file name
    """
    if output_dir is None:
        output_dir = config.tec_dir
    mlat_vals = (mlat_bins[:-1] + mlat_bins[1:]) / 2
    mlt_vals = (mlt_bins[:-1] + mlt_bins[1:]) / 2
    bins = [mlat_bins, mlt_bins]
    years = np.arange(start_year, end_year, apex_dt)
    t0 = time.time()
    for year in years:
        process_multiple_files(year, min(end_year, year + apex_dt), bins, dt=file_dt, map_period=map_period,
                               output_dir=output_dir, file_name_pattern=file_name_pattern)
    tf = time.time()
    print((tf - t0) / 60)
    # make grid file
    io.write_h5(os.path.join(output_dir, "grid.h5"), mlt=mlt_vals, mlat=mlat_vals)
