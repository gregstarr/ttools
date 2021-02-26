import numpy as np
import h5py
import os
import glob
import apexpy

from ttools import utils, io
from ttools.config import mlt_vals


for year in range(2010, 2022):
    apex = apexpy.Apex(year)
    year_floor = np.datetime64(f"{year}-01-01")
    for month in range(1, 13):
        print(f"{year} - {month}")
        start = np.datetime64(f"{year}-{month:02d}")
        end = start + np.timedelta64(1, 'M')
        start_doy = (start - year_floor).astype('timedelta64[D]').astype(int) + 1
        end_doy = (end - year_floor).astype('timedelta64[D]').astype(int) + 1
        f_year = []
        f_doy = []
        f_time = []
        alt = []
        glat = []
        glon = []
        for doy in range(start_doy - 1, end_doy + 1):
            files = glob.glob(os.path.join(f"E:\\dmsp\\17\\edr-aurora\\{year:4d}\\{doy:03d}", "*.nc"))
            for fn in files:
                with h5py.File(fn, 'r') as f:
                    f_year.append(f['YEAR'][()] - 1970)
                    f_doy.append(f['DOY'][()])
                    f_time.append(f['TIME'][()])
                    alt.append(f['ALTITUDE'][()])
                    glat.append(f['MODEL_NORTH_GEOGRAPHIC_LATITUDE'][()])
                    glon.append(f['MODEL_NORTH_GEOGRAPHIC_LONGITUDE'][()])
        times = np.array(f_year, dtype='datetime64[Y]') + (np.array(f_doy, dtype='timedelta64[D]') - 1) + np.array(f_time, dtype='timedelta64[s]')
        if times.size == 0:
            continue
        bad_ind, = np.where(np.diff(times).astype(float) < 0)
        times[bad_ind] = times[bad_ind - 1] + (times[bad_ind + 1] - times[bad_ind - 1]) / 2
        times[-1] = times[-2] + np.median(np.diff(times))
        mean_alt = np.array([a.mean() for a in alt])
        mlat = np.empty((times.shape[0], mlt_vals.shape[0]))
        for i, (lat, lon, height, t) in enumerate(zip(glat, glon, mean_alt, times)):
            apx_lat, apx_lon = apex.geo2apex(lat, lon, height)
            mlt = apex.mlon2mlt(apx_lon, utils.datetime64_to_datetime(t))
            mlat[i] = np.interp(mlt_vals, mlt, apx_lat, period=24)
        outfn = f"E:\\auroral_boundary\\{year:4d}_{month:02d}_arb.h5"
        merged_times = times.astype('datetime64[s]').astype(int)
        merged_mlat = mlat
        if os.path.exists(outfn):
            existing_mlat, existing_times = io.open_arb_file(outfn)
            merged_times = np.concatenate((existing_times, times.astype('datetime64[s]').astype(int)))
            merged_mlat = np.concatenate((existing_mlat, mlat), axis=0)
            sort_idx = np.argsort(merged_times)
            merged_times = merged_times[sort_idx]
            merged_mlat = merged_mlat[sort_idx, :]
        io.write_h5(outfn, times=merged_times, mlat=merged_mlat)
