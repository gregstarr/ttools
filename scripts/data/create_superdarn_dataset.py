import numpy as np
import pandas
import glob
import apexpy
from scipy import stats

from ttools import io


if __name__ == "__main__":

    for year in range(2010, 2021):
        year_floor = np.datetime64(f"{year}-01-01")
        apex = apexpy.Apex(year)
        for month in range(1, 13):
            print(f"{year} - {month}")
            start = np.datetime64(f"{year}-{month:02d}")
            end = start + np.timedelta64(1, 'M')
            bins = [np.arange(start, end + np.timedelta64(1, 'h'), np.timedelta64(1, 'h')).astype('datetime64[s]').astype(float), np.arange(29.5, 90), np.arange(-12, 12 + 24 / 360, 48 / 360)]
            outfn = f"E:\\superdarn\\{year:4d}_{month:02d}_superdarn.h5"
            superdarn_files = glob.glob(f"E:\\superdarn\\raw\\{year:4d}{month:02d}*_north.csv")
            sd_time = []
            sd_mlat = []
            sd_mlon = []
            sd_kvect = []
            sd_vel = []
            for file in superdarn_files:
                sd_data = pandas.read_csv(file, skiprows=14)
                sd_time.append(pandas.to_datetime(sd_data['time']).values.astype('datetime64[s]'))
                sd_mlat.append(sd_data['vector_mlat'].values.astype(float))
                sd_mlon.append(sd_data['vector_mlon'].values.astype(float))
                sd_kvect.append(sd_data['vector_kvect'].values.astype(float))
                sd_vel.append(sd_data['vector_vel_median'].values.astype(float))
            sd_time = np.concatenate(sd_time)
            sd_mlat = np.concatenate(sd_mlat)
            sd_mlon = np.concatenate(sd_mlon)
            sd_kvect = np.concatenate(sd_kvect)
            sd_vel = np.concatenate(sd_vel)
            mlt = apex.mlon2mlt(sd_mlon, sd_time)
            mlt[mlt > 12] -= 24
            sd_theta = np.pi + np.pi * (mlt - 6) / 12 - np.deg2rad(sd_kvect)
            sd_fx = np.cos(sd_theta) * sd_vel
            sd_fy = np.sin(sd_theta) * sd_vel
            time = sd_time.astype('datetime64[s]').astype(float)
            sample = np.column_stack((time, sd_mlat, mlt))
            fx = stats.binned_statistic_dd(sample, sd_fx, 'mean', bins).statistic
            fy = stats.binned_statistic_dd(sample, sd_fy, 'mean', bins).statistic
            count = stats.binned_statistic_dd(sample, None, 'count', bins).statistic
            fx[count < 2] = np.nan
            fy[count < 2] = np.nan
            io.write_h5(outfn, time=bins[0][:-1], fx=fx, fy=fy)
