import os
import h5py
import numpy as np


# madrigal TEC data directory
madrigal_dir = "E:\\tec_data\\download"
madrigal_lat = np.arange(-90, 90)
madrigal_lon = np.arange(-180, 180)

# processed TEC data directory
tec_dir = "E:\\tec_data"

# SWARM data directory
swarm_dir = "E:\\swarm\\extracted"
swarm_coords_dir = "E:\\swarm\\coordinates"

grid_file = os.path.join(os.path.dirname(__file__), "grid.h5")
with h5py.File(grid_file, 'r') as f:
    mlt_vals = f['mlt'][()]
    mlat_vals = f['mlat'][()]
mlt_grid, mlat_grid = np.meshgrid(mlt_vals, mlat_vals)
mlt_grid = mlt_grid.astype(np.float32)
mlat_grid = mlat_grid.astype(np.float32)

kp_file = os.path.join(os.path.dirname(__file__), "2000_2020_kp_ap.txt")

# other settings
PARALLEL = True


if __name__ == "__main__":
    year = 2012
    month = 6
    print(tec_file_pattern.format(year=year, month=month))
