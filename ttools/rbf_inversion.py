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
import pandas
import multiprocessing

from skimage.util import view_as_windows
from skimage import measure
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix
from scipy.interpolate import RectSphereBivariateSpline

from ttools import utils, trough_model, config, convert


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
    """Use a moving average filter to estimate the background TEC value. `patch_shape` must contain odd numbers

    Parameters
    ----------
    tec: numpy.ndarray[float]
    patch_shape: tuple

    Returns
    -------
    numpy.ndarray[float]
    """
    assert all([2 * (p // 2) + 1 == p for p in patch_shape]), "patch_shape must be all odd numbers"
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


def get_optimization_args(x, times, mlt_vals=config.mlt_vals, mlat_grid=config.mlat_grid, model_weight_max=25,
                          rbf_bw=1, tv_hw=2, tv_vw=1, l2_weight=.1, tv_weight=.06, prior_order=1, prior='empirical',
                          arb=None, arb_offset=-1):
    all_args = []
    # get rbf basis matrix
    basis = get_rbf_matrix(x.shape[1:], rbf_bw)
    # get tv matrix
    tv = get_tv_matrix(x.shape[1:], tv_hw, tv_vw) * tv_weight
    if prior == 'empirical_model':
        # get deminov model
        ut = times.astype('datetime64[s]').astype(float)
        model_mlat = trough_model.get_model(ut, mlt_vals)
    elif prior == 'auroral_boundary':
        model_mlat = arb + arb_offset
    else:
        raise Exception("Invalid prior name")
    for i in range(times.shape[0]):
        # l2 norm cost away from model
        if prior_order == 1:
            l2 = abs(mlat_grid - model_mlat[i, :])
        elif prior_order == 2:
            l2 = (mlat_grid - model_mlat[i, :]) ** 2
        else:
            raise Exception("Invalid prior order")
        l2 -= l2.min()
        l2 = (model_weight_max - 1) * l2 / l2.max() + 1
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
    try:
        prob.solve(solver=cp.GUROBI)
    except Exception as e:
        try:
            print("GUROBI FAILED, RUNNING AGAIN VERBOSE:", e)
            prob.solve(solver=cp.GUROBI, verbose=True)
        except Exception as e:
            print("GUROBI FAILED, USING ECOS:", e)
            prob.solve(solver=cp.ECOS)
    return u.value.reshape(output_shape)


def run_multiple(args, parallel=True):
    if parallel:
        with multiprocessing.Pool(processes=4) as p:
            results = p.starmap(run_single, args)
    else:
        results = []
        for arg in args:
            results.append(run_single(*arg))
    return np.stack(results, axis=0)


def get_artifacts(mlt_grid, ssmlon, artifact_key, fn="E:\\tec_data\\tec_artifact.npz"):
    artifacts = np.load(fn)
    mlon = convert.mlt_to_mlon_sub(mlt_grid[None, :, :], ssmlon[:, None, None]) * np.pi / 180
    comlat = (90 - config.mlat_grid) * np.pi / 180

    sp = RectSphereBivariateSpline((90 - artifacts['mlat_vals'][::-1]) * np.pi / 180,
                                   artifacts['mlon_vals'] * np.pi / 180,
                                   artifacts[artifact_key][::-1, :])

    corr = np.empty_like(mlon)
    for i in range(ssmlon.shape[0]):
        corr[i] = sp(comlat.ravel(), mlon[i].ravel(), grid=False).reshape(comlat.shape)[::-1, :]
    return corr


def fix_boundaries(labels):
    fixed = labels.copy()
    while True:
        boundary_pairs = np.unique(fixed[:, [0, -1]], axis=0)
        if np.all((boundary_pairs[:, 0] == boundary_pairs[:, 1]) | np.any(boundary_pairs == 0, axis=1)):
            break
        for i in range(boundary_pairs.shape[0]):
            if np.any(boundary_pairs[i] == 0) or boundary_pairs[i, 0] == boundary_pairs[i, 1]:
                continue
            fixed[fixed == boundary_pairs[i, 1]] = boundary_pairs[i, 0]
            break
    return fixed


def postprocess(initial_trough, perimeter_th=50, area_th=1, arb=None):
    trough = initial_trough.copy()
    for t in range(trough.shape[0]):
        tmap = trough[t]
        labeled = measure.label(tmap, connectivity=2)
        labeled = fix_boundaries(labeled)
        props = pandas.DataFrame(measure.regionprops_table(labeled, properties=('label', 'area', 'perimeter')))
        error_mask = (props['perimeter'] < perimeter_th) | (props['area'] < area_th)
        for i, r in props[error_mask].iterrows():
            tmap[labeled == r['label']] = 0
        trough[t] = tmap
    if arb is not None:
        for t in range(trough.shape[0]):
            trough[t] *= (config.mlat_grid < arb[t, None, :])
    return trough


def get_tec_troughs(tec, input_times, bg_est_shape=(3, 15, 15), model_weight_max=20, rbf_bw=1, tv_hw=1, tv_vw=1,
                    l2_weight=.1, tv_weight=.05, perimeter_th=50, area_th=20, artifact_correction=None,
                    arb=None, prior_order=1, prior='empirical', prior_arb_offset=-1):
    # preprocess
    print("Preprocessing TEC data")
    x, times = preprocess_interval(tec, input_times, bg_est_shape=bg_est_shape)
    if artifact_correction is not None:
        x -= artifact_correction
    # setup optimization
    print("Setting up inversion optimization")
    args = get_optimization_args(x, times, model_weight_max=model_weight_max, rbf_bw=rbf_bw, tv_hw=tv_hw, tv_vw=tv_vw,
                                 l2_weight=l2_weight, tv_weight=tv_weight, prior_order=prior_order, prior=prior,
                                 arb=arb, arb_offset=prior_arb_offset)
    # run optimization
    print("Running inversion optimization")
    model_output = run_multiple(args, parallel=config.PARALLEL)
    # threshold
    initial_trough = model_output >= 1
    # postprocess
    print("Postprocessing inversion results")
    trough = postprocess(initial_trough, perimeter_th, area_th, arb)
    return trough, x
