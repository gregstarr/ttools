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
        outfn = f"E:\\dmsp_flow2\\{year:4d}_{month:02d}_dmsp_flow.h5"
        keys = ['ut1_unix', 'gdlat', 'glon', 'gdalt', 'kindat', 'ni', 'ni_idm', 'ion_v_sat_for', 'ion_v_sat_left',
                'vert_ion_v', 'ion_v_for_flag', 'ion_v_left_flag', 'ion_v_up_flag', 'ni_rpa_flag', 'idm_flag_ut']
        data = {key: [] for key in keys}
        for day in range(1, 32):
            print(day)
            files = glob.glob(os.path.join(f"E:\\dmsp_flow2\\download\\dms_ut_{year:04d}{month:02d}{day:02d}*"))
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
        for sat in np.unique(data['kindat']):
            sat_mask = data['kindat'] == sat
            time_mask = np.in1d(ut, data['ut1_unix'][sat_mask])
            mlat, mlt = apex.convert(data['gdlat'][sat_mask], data['glon'][sat_mask], 'geo', 'mlt', height=np.nanmean(data['gdalt'][sat_mask]), datetime=times[sat_mask])
            write_dict[f'/dmsp{sat}/mlat'] = np.ones_like(ut, dtype=float) * np.nan
            write_dict[f'/dmsp{sat}/mlat'][time_mask] = mlat
            write_dict[f'/dmsp{sat}/mlt'] = np.ones_like(ut, dtype=float) * np.nan
            write_dict[f'/dmsp{sat}/mlt'][time_mask] = mlt
            for key in ['kindat', 'ni', 'ni_idm', 'ion_v_sat_for', 'ion_v_sat_left', 'vert_ion_v', 'ion_v_for_flag',
                        'ion_v_left_flag', 'ion_v_up_flag', 'ni_rpa_flag', 'idm_flag_ut']:
                write_dict[f'/dmsp{sat}/{key}'] = np.ones_like(ut, dtype=float) * np.nan
                write_dict[f'/dmsp{sat}/{key}'][time_mask] = data[key][sat_mask]
        io.write_h5(outfn, **write_dict)
