import numpy as np

from ttools import utils, io, old, satellite


for year in range(2013, 2020):
    year_floor = np.datetime64(f"{year}-01-01")
    for month in range(1, 13):
        print(f"{year} - {month}")
        start = np.datetime64(f"{year}-{month:02d}")
        end = start + np.timedelta64(1, 'M')
        outfn = f"E:\\swarm\\{year:4d}_{month:02d}_swarm.h5"
        keys = ['apex_lat', 'apex_lon', 'mlt', 'n']
        write_dict = {}
        for sat in satellite.SATELLITES['swarm']:
            data, times = old.get_swarm_data(start, end, sat)
            for key in keys:
                write_dict[f'/swarm{sat}/{key}'] = data[key]
        if all(np.isnan(write_dict[key]).all() for key in write_dict):
            print("NO DATA")
            continue
        write_dict['ut_ms'] = times.astype(float)
        io.write_h5(outfn, **write_dict)