import os
import h5py
import numpy as np
import cvxpy as cp


base_dir = madrigal_dir = madrigal_lat = madrigal_lon = tec_dir = arb_dir = swarm_dir = swarm_coords_dir = kp_file \
    = grid_file = mlt_vals = mlat_vals = mlt_grid = mlat_grid = PARALLEL = artifact_file = SOLVER = omni_file \
    = dmsp_dir = superdarn_dir = None


def update(_base_dir="E:\\"):
    global base_dir, madrigal_dir, madrigal_lat, madrigal_lon, tec_dir, arb_dir, swarm_dir, swarm_coords_dir
    global kp_file, grid_file, mlt_vals, mlat_vals, mlt_grid, mlat_grid, PARALLEL, artifact_file, SOLVER, omni_file
    global dmsp_dir, superdarn_dir
    base_dir = _base_dir

    # madrigal TEC data directory
    madrigal_dir = os.path.join(base_dir, "tec_data", "download")
    madrigal_lat = np.arange(-90, 90)
    madrigal_lon = np.arange(-180, 180)

    # processed TEC data directory
    tec_dir = os.path.join(base_dir, "tec_data")
    artifact_file = os.path.join(tec_dir, 'tec_artifact.npz')

    # processed auroral boundary data directory
    arb_dir = os.path.join(base_dir, "auroral_boundary")

    # processed auroral boundary data directory
    dmsp_dir = os.path.join(base_dir, "dmsp_flow")

    # SWARM data directory
    swarm_dir = os.path.join(base_dir, "swarm")
    swarm_coords_dir = os.path.join(swarm_dir, 'coords')

    superdarn_dir = os.path.join(base_dir, 'superdarn')

    grid_file = os.path.join(tec_dir, "grid.h5")
    if os.path.exists(grid_file):
        with h5py.File(grid_file, 'r') as f:
            mlt_vals = f['mlt'][()]
            mlat_vals = f['mlat'][()]
        mlt_grid, mlat_grid = np.meshgrid(mlt_vals, mlat_vals)
    else:
        print("GRID FILE NOT FOUND")
        mlt_vals = mlat_vals = mlt_grid = mlat_grid = None

    kp_file = os.path.join(base_dir, "2000_2020_kp_ap.txt")
    omni_file = os.path.join(base_dir, "omni2_all_years.dat")

    # other settings
    PARALLEL = True
    SOLVER = cp.GUROBI


update()
