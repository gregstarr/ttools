"""
steps:
    - get data interval
    - preprocess
        - log, background estimation, downsampling
    - setup and run optimization problem
    - threshold at 1

Example:
```
# load data
tec, times, ssmlon, n = io.get_tec_data(start_time, end_time)
# preprocess
x, times = rbf_inversion.preprocess_interval(tec, times)
# setup optimization
args = rbf_inversion.get_optimization_args(x, times)
# run optimization
model_output = rbf_inversion.run_multiple(args)
# threshold
initial_trough = model_output >= 1
# postprocess
trough = rbf_inversion.postprocess(initial_trough)
```
"""
import numpy as np
import bottleneck as bn
import cvxpy as cp
import warnings
import pandas
import multiprocessing

from skimage.util import view_as_windows
from skimage import measure
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix

from ttools import utils, trough_model, config


DEFAULT_PARAMS = {
    'tv_weight': .06,
    'l2_weight': .05,
    'bge_spatial_size': 17,
    'bge_temporal_rad': 1,
    'cos_mlat': True,
    'rbf_bw': 1,
    'tv_hw': 1,
    'tv_vw': 1,
    'model_weight_max': 10,
    'perimeter_th': 50
}


def extract_patches(arr, patch_shape, step=1):
    """Assuming `arr` is 3D (time, lat, lon). `arr` will be padded, then have patches extracted using
    `skimage.util.view_as_windows`. The padding will be "edge" for lat, and "wrap" for lon, with no padding for
    time. Returned array will have same lat and lon dimension length as input and a different time dimension length
    depending on `patch_shape`.

    Parameters
    ----------
    arr: numpy.ndarray
        must be 3 dimensional
    patch_shape: tuple
        must be length 3
    step: int
    Returns
    -------
    patches view of padded array
        shape (arr.shape[0] - patch_shape[0] + 1, arr.shape[1], arr.shape[2]) + patch_shape
    """
    assert arr.ndim == 3 and len(patch_shape) == 3, "Invalid input args"
    # lat padding
    padded = np.pad(arr, ((0, 0), (patch_shape[1] // 2, patch_shape[1] // 2), (0, 0)), 'edge')
    # lon padding
    padded = np.pad(padded, ((0, 0), (0, 0), (patch_shape[2] // 2, patch_shape[2] // 2)), 'wrap')
    patches = view_as_windows(padded, patch_shape, step)
    return patches


def estimate_background(tec, patch_shape):
    """

    Parameters
    ----------
    tec
    patch_shape

    Returns
    -------

    """
    patches = extract_patches(tec, patch_shape)
    return bn.nanmean(patches.reshape((tec.shape[0] - patch_shape[0] + 1, ) + tec.shape[1:] + (-1, )), axis=-1)


def preprocess_interval(tec, times, min_val=0, max_val=150, bg_est_shape=(3, 15, 15), ds=None, n_samples=None):
    tec = tec.copy()
    # throw away outlier values
    tec[tec > max_val] = np.nan
    tec[tec < min_val] = np.nan
    # change to log
    log_tec = np.log10(tec + .001)
    # estimate background
    bg = estimate_background(log_tec, bg_est_shape)
    # subtract background
    log_tec, t, = utils.moving_func_trim(bg_est_shape[0], log_tec, times)
    x = log_tec - bg
    # downsample
    if ds is not None:
        print(n_samples)
        raise NotImplementedError
    return x, t


def get_rbf_matrix(shape, bandwidth=1):
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xy = np.column_stack((X.ravel(), Y.ravel()))
    gamma = np.log(2) / bandwidth**2
    basis = rbf_kernel(xy, xy, gamma)
    basis[basis < .01] = 0
    return csr_matrix(basis)


def get_tv_matrix(im_shape, hw=1, vw=1):
    size = im_shape[0] * im_shape[1]

    right = np.eye(size)
    cols = im_shape[1] * (np.arange(size) // im_shape[1]) + (np.arange(size) + 1) % im_shape[1]
    right[np.arange(size), cols] = -1

    left = np.eye(size)
    cols = im_shape[1] * (np.arange(size) // im_shape[1]) + (np.arange(size) - 1) % im_shape[1]
    left[np.arange(size), cols] = -1

    up = np.eye(size)
    cols = np.arange(size) + im_shape[1]
    mask = cols < size
    cols = cols[mask]
    up[np.arange(size)[mask], cols] = -1

    down = np.eye(size)
    cols = np.arange(size) - im_shape[1]
    mask = cols >= 0
    cols = cols[mask]
    down[np.arange(size)[mask], cols] = -1
    return csr_matrix(hw * (right + left) + vw * (up + down))


def get_optimization_args(x, times, mlt_vals=config.mlt_vals, mlat_grid=config.mlat_grid, model_weight_max=30,
                          rbf_bw=1, tv_hw=1, tv_vw=1, l2_weight=.15, tv_weight=.06):
    all_args = []
    # get deminov model
    ut = times.astype('datetime64[s]').astype(float)
    model_mlat = trough_model.get_model(ut, mlt_vals)
    # get rbf basis matrix
    basis = get_rbf_matrix(x.shape[1:], rbf_bw)
    # get tv matrix
    tv = get_tv_matrix(x.shape[1:], tv_hw, tv_vw) * tv_weight
    for i in range(times.shape[0]):
        # l2 norm cost away from model
        l2 = (mlat_grid - model_mlat[i, :]) ** 2
        l2 /= (l2.max() / model_weight_max)
        l2 += 1
        l2 *= l2_weight
        fin_mask = np.isfinite(x[i].ravel())
        args = (cp.Variable(x.shape[1] * x.shape[2]), basis[fin_mask, :], x[i].ravel()[fin_mask], tv, l2.ravel(),
                times[i], mlat_grid.shape)
        all_args.append(args)
    return all_args


def run_single(u, basis, x, tv, l2, t, output_shape):
    print(t)
    main_cost = u.T @ basis.T @ x
    tv_cost = cp.norm1(tv @ u)
    l2_cost = l2 @ (u ** 2)
    total_cost = main_cost + tv_cost + l2_cost
    prob = cp.Problem(cp.Minimize(total_cost), [u >= 0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        prob.solve(solver=cp.SCS, max_iters=2000)
    return u.value.reshape(output_shape)


def run_multiple(args):
    with multiprocessing.Pool(processes=8) as p:
        results = p.starmap(run_single, args)
    return np.stack(results, axis=0)


def fix_boundaries(labels):
    fixed = labels.copy()
    while True:
        boundary_pairs = np.unique(fixed[:, [0, -1]], axis=0)
        if np.all(boundary_pairs[:, 0] == boundary_pairs[:, 1]):
            break
        for i in range(boundary_pairs.shape[0]):
            if np.any(boundary_pairs[i] == 0) or boundary_pairs[i, 0] == boundary_pairs[i, 1]:
                continue
            fixed[fixed == boundary_pairs[i, 1]] = boundary_pairs[i, 0]
            break
    return fixed


def postprocess(initial_trough, perimeter_th=50, area_th=1):
    trough = initial_trough.copy()
    for t in range(trough.shape[0]):
        tmap = trough[t]
        labeled = measure.label(tmap, connectivity=2)
        labeled = fix_boundaries(labeled)
        props = pandas.DataFrame(measure.regionprops_table(labeled, properties=('label', 'area', 'perimeter')))
        error_mask = (props['perimeter'] < perimeter_th) + (props['area'] < area_th)
        for i, r in props[error_mask].iterrows():
            tmap[labeled == r['label']] = 0
        trough[t] = tmap
    return trough
