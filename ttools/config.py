import os
import h5py
import numpy as np


# madrigal TEC data directory
madrigal_dir = "E:\\tec_data\\download"
madrigal_lat = np.arange(-90, 90)
madrigal_lon = np.arange(-180, 180)

# processed TEC data directory
tec_dir = "E:\\tec_data"

# processed auroral boundary data directory
arb_dir = "E:\\auroral_boundary"

# SWARM data directory
swarm_dir = "E:\\swarm\\extracted"
swarm_coords_dir = "E:\\swarm\\coordinates"

grid_file = os.path.join(os.path.dirname(__file__), "grid.h5")
with h5py.File(grid_file, 'r') as f:
    mlt_vals = f['mlt'][()]
    mlat_vals = f['mlat'][()]
mlt_grid, mlat_grid = np.meshgrid(mlt_vals, mlat_vals)

kp_file = os.path.join(os.path.dirname(__file__), "2000_2020_kp_ap.txt")

# other settings
PARALLEL = True
