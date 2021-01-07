import numpy as np
import apexpy
from scipy.stats import binned_statistic_2d
import time
from multiprocessing import Pool

from ttools import io, config, convert, utils


def assemble_binning_args(mlat, mlt, tec, times, bins, map_period):
    """Creates a list of tuple arguments to be passed to `calculate_bins`. `calculate_bins` is called by the process
    pool manager using each tuple in the list returned by this function as arguments. Each set of arguments corresponds
    to one processed TEC map and should span a time period specified by `map_period`. `map_period` should evenly divide
    24h, and probably should be a multiple of 5 min.

    Parameters
    ----------
    mlat: numpy.ndarray[float]
    mlt: numpy.ndarray[float]
    tec: numpy.ndarray[float]
    times: numpy.ndarray[float]
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
    while current_time <= times[-1]:
        start = np.argmax(times >= current_time)
        end = np.argmax(times >= current_time + map_period)
        if end == 0:
            end = times.shape[0]
        time_slice = slice(start, end)
        fin_mask = np.isfinite(tec[time_slice])
        mlat_r = mlat[time_slice][fin_mask].copy()
        mlt_r = mlt[time_slice][fin_mask].copy()
        tec_r = tec[time_slice][fin_mask].copy()
        args.append((mlat_r, mlt_r, tec_r, times[time_slice], bins))
        current_time += map_period
    return args


def calculate_bins(mlat, mlt, tec, times, bins):
    """Calculates TEC in MLAT - MLT bins. Executed in process pool.

    Parameters
    ----------
    mlat: numpy.ndarray[float] (N, )
    mlt: numpy.ndarray[float] (N, )
    tec: numpy.ndarray[float] (N, )
    times: numpy.ndarray[float] (T, )
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
    return times[0], final_tec, final_tec_n, final_tec_s


def process_month(start_date, mlat_grid, mlon_grid, converter, bins, map_period=np.timedelta64(1, 'h'),
                  madrigal_dir=None):
    """Processes one month's worth of madrigal data. Opens madrigal h5 files, converts input mlon grid to MLT by
    computing subsolar points at each time step, sets up and runs TEC binning, unpacks and returns results.

    Parameters
    ----------
    start_date: np.datetime64
    mlat_grid: numpy.ndarray[float] (X, Y)
    mlon_grid: numpy.ndarray[float] (X, Y)
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
    print(start_date)
    if start_date.dtype != np.dtype('datetime64[M]'):
        start_date = start_date.astype('datetime64[M]')
    tec, ts = io.get_madrigal_data(start_date, start_date + 1, dir=madrigal_dir)
    print("Converting coordinates")
    mlat, mlt, ssmlon = convert.mlon_to_mlt_grid(mlat_grid, mlon_grid, ts, converter)
    print("Setting up for binning")
    args = assemble_binning_args(mlat, mlt, tec, ts, bins, map_period)
    print(f"Calculating bins for {len(args)} time steps")
    with Pool(processes=8) as p:
        pool_result = p.starmap(calculate_bins, args)
    print("Calculated bins")
    times = np.array([r[0] for r in pool_result])
    tec = np.array([r[1] for r in pool_result])
    n = np.array([r[2] for r in pool_result])
    std = np.array([r[3] for r in pool_result])
    return times, tec, n, std, ssmlon


def get_mag_grid(ref_lat, ref_lon, converter):
    lon_grid, lat_grid = np.meshgrid(ref_lon, ref_lat)
    mlat, mlon = converter.convert(lat_grid.ravel(), lon_grid.ravel(), 'geo', 'apex', height=350)
    mlat = mlat.reshape(lat_grid.shape)
    mlon = mlon.reshape(lat_grid.shape)
    return mlat, mlon


def process_year(start_date, ref_lat, ref_lon, bins):
    """Processes monthly madrigal data and writes to files.

    Parameters
    ----------
    start_date: np.datetime64
    ref_lat: numpy.ndarray[float]
        latitude values of madrigal lat-lon grid
    ref_lon: numpy.ndarray[float]
        longitude values of madrigal lat-lon grid
    bins: list[numpy.ndarray[float] (X + 1, ), numpy.ndarray[float] (Y + 1, )]
    """
    if start_date.dtype != np.dtype('datetime64[Y]'):
        start_date = start_date.astype('datetime64[Y]')
    apex_date = utils.datetime64_to_datetime(start_date)
    converter = apexpy.Apex(date=apex_date)
    mlat, mlon = get_mag_grid(ref_lat, ref_lon, converter)
    months = np.arange(start_date, start_date + 1, np.timedelta64(1, 'M'))
    for month in months:
        times, tec, n, std, ssmlon = process_month(month, mlat, mlon, converter, bins)
        if np.isfinite(tec).any():
            ymd = utils.decompose_datetime64(month)
            fn = config.tec_file_pattern.format(year=ymd[0, 0], month=ymd[0, 1])
            io.write_file(fn, times=times.astype('datetime64[s]').astype(int), tec=tec, n=n, std=std, ssmlon=ssmlon)
        else:
            print(f"No data for month: {month}, not writing file")


def process_dataset(start_year, end_year, mlat_bins, mlt_bins):
    """'main' function for creating tec dataset. Brief setup, then calls `process_year` on each year as specified by
    inputs. Writes grid file.

    Parameters
    ----------
    start_year: np.datetime64
    end_year: np.datetime64
    mlat_bins: np.ndarrat[float]
    mlt_bins: np.ndarrat[float]
    """
    mlat_vals = (mlat_bins[:-1] + mlat_bins[1:]) / 2
    mlt_vals = (mlt_bins[:-1] + mlt_bins[1:]) / 2
    bins = [mlat_bins, mlt_bins]
    years = np.arange(start_year, end_year, np.timedelta64(1, 'Y'))
    t0 = time.time()
    for year in years:
        process_year(year, config.madrigal_lat, config.madrigal_lon, bins)
    tf = time.time()
    print((tf - t0) / 60)
    # make grid file
    io.write_file(config.grid_file, mlt=mlt_vals, mlat=mlat_vals)


if __name__ == "__main__":
    import sys
    start_year = "2010"
    end_year = "2021"
    if len(sys.argv) > 1:
        start_year = sys.argv[1]
        end_year = sys.argv[2]
    # configure date ranges
    START_YEAR = np.datetime64(start_year)
    END_YEAR = np.datetime64(end_year)

    # configure grid
    MLAT_BINS = np.arange(29.5, 90)
    MLT_BINS = np.arange(-12, 12 + 24 / 360, 48 / 360)
    process_dataset(START_YEAR, END_YEAR, MLAT_BINS, MLT_BINS)