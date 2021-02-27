import os
import h5py
import numpy as np


def update(_base_dir="E:\\"):
    global base_dir, madrigal_dir, madrigal_lat, madrigal_lon, tec_dir, arb_dir, swarm_dir, swarm_coords_dir
    global kp_file, mlt_grid, mlat_grid, PARALLEL
    base_dir = _base_dir

    # madrigal TEC data directory
    madrigal_dir = os.path.join(base_dir, "tec_data", "download")
    madrigal_lat = np.arange(-90, 90)
    madrigal_lon = np.arange(-180, 180)

    # processed TEC data directory
    tec_dir = os.path.join(base_dir, "tec_data")

    # processed auroral boundary data directory
    arb_dir = os.path.join(base_dir, "auroral_boundary")

    # SWARM data directory
    swarm_dir = os.path.join(base_dir, "swarm", "extracted")
    swarm_coords_dir = os.path.join(base_dir, "swarm", "coordinates")

    grid_file = os.path.join(tec_dir, "grid.h5")
    if os.path.exists(grid_file):
        with h5py.File(grid_file, 'r') as f:
            mlt_vals = f['mlt'][()]
            mlat_vals = f['mlat'][()]
        mlt_grid, mlat_grid = np.meshgrid(mlt_vals, mlat_vals)
    else:
        print("GRID FILE NOT FOUND")

    kp_file = os.path.join(base_dir, "2000_2020_kp_ap.txt")

    # other settings
    PARALLEL = True


update()
