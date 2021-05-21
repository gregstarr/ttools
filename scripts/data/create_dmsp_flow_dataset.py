import numpy as np
import h5py
import os
import glob
import apexpy

from ttools import io


for year in range(2010, 2022):
    apex = apexpy.Apex(year)
    year_floor = np.datetime64(f"{year}-01-01")
    for month in range(1, 13):
        print(f"{year} - {month}")
        outfn = f"E:\\dmsp_flow\\{year:4d}_{month:02d}_dmsp_flow.h5"
        keys = ['ut1_unix', 'gdlat', 'glon', 'gdalt', 'sat_id', 'ne', 'hor_ion_v', 'vert_ion_v']
        data = {key: [] for key in keys}
        for day in range(1, 32):
            print(day)
            files = glob.glob(os.path.join(f"H:\\dmsp_flow\\dms_{year:04d}{month:02d}{day:02d}*"))
            for fn in files:
                with h5py.File(fn, 'r') as f:
                    for key in keys:
                        data[key].append(f['Data/Table Layout'][key])
        print()
        if len(data['ut1_unix']) == 0:
            continue
        for key in keys:
            data[key] = np.concatenate(data[key], axis=0)
        times = np.datetime64('1970-01-01') + data['ut1_unix'].astype('timedelta64[s]')
        ut = np.unique(data['ut1_unix'])
        write_dict = {'ut': ut}
        for sat in np.unique(data['sat_id']):
            sat_mask = data['sat_id'] == sat
            time_mask = np.in1d(ut, data['ut1_unix'][sat_mask])
            mlat, mlt = apex.convert(data['gdlat'][sat_mask], data['glon'][sat_mask], 'geo', 'mlt', height=np.nanmean(data['gdalt'][sat_mask]), datetime=times[sat_mask])
            write_dict[f'/dmsp{sat}/mlat'] = np.ones_like(ut, dtype=float) * np.nan
            write_dict[f'/dmsp{sat}/mlat'][time_mask] = mlat
            write_dict[f'/dmsp{sat}/mlt'] = np.ones_like(ut, dtype=float) * np.nan
            write_dict[f'/dmsp{sat}/mlt'][time_mask] = mlt
            write_dict[f'/dmsp{sat}/ne'] = np.ones_like(ut, dtype=float) * np.nan
            write_dict[f'/dmsp{sat}/ne'][time_mask] = data['ne'][sat_mask]
            write_dict[f'/dmsp{sat}/hor_ion_v'] = np.ones_like(ut, dtype=float) * np.nan
            write_dict[f'/dmsp{sat}/hor_ion_v'][time_mask] = data['hor_ion_v'][sat_mask]
            write_dict[f'/dmsp{sat}/vert_ion_v'] = np.ones_like(ut, dtype=float) * np.nan
            write_dict[f'/dmsp{sat}/vert_ion_v'][time_mask] = data['vert_ion_v'][sat_mask]
        io.write_h5(outfn, **write_dict)
