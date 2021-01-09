import os
import h5py
import numpy as np


# madrigal TEC data directory
madrigal_dir = "E:\\tec_data\\download"
madrigal_lat = np.arange(-90, 90)
madrigal_lon = np.arange(-180, 180)

# processed TEC data directory
tec_dir = "E:\\tec_data"
tec_file_pattern = os.path.join(tec_dir, "{year:04d}_{month:02d}_tec.h5")

# SWARM data directory
swarm_dir = "E:\\swarm\\extracted"
swarm_coords_dir = "E:\\swarm\\coordinates"

grid_file = os.path.join(tec_dir, "grid.h5")
# with h5py.File(grid_file, 'r') as f:
#     mlt_vals = f['mlt'][()]
#     mlat_vals = f['mlat'][()]
# mlt_grid, mlat_grid = np.meshgrid(mlt_vals, mlat_vals)
# theta_vals = np.pi * (mlt_vals - 6) / 12
# radius_vals = 90 - mlat_vals
# theta_grid, radius_grid = np.meshgrid(theta_vals, radius_vals)


if __name__ == "__main__":
    year = 2012
    month = 6
    print(tec_file_pattern.format(year=year, month=month))
