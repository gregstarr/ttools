"""
steps:
    - preprocess
    - setup and run optimization problem
    - threshold at 1
"""
import numpy as np
import h5py
import os
import bottleneck as bn
import cvxpy as cp
import warnings
import pandas

from skimage.util import view_as_windows
from skimage import morphology, measure
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix
from scipy.interpolate import interp2d

from ttools import utils, trough_model


def get_datetime64_year_month(datetime):
    year = datetime.astype('datetime64[Y]').astype(int) + 1970
    month = datetime.astype('datetime64[M]').astype(int) % 12 + 1
    return year, month


def get_tec_maps(year, month, base_dir="E:\\tec_data"):
    fn = os.path.join(base_dir, f"{year:04d}_{month:02d}_tec.h5")
    with h5py.File(fn, 'r') as f:
        tec = f['tec'][()]
        labels = f['labels'][()]
        start_time = f['start_time'][()]
    return tec, labels, start_time


def polar_pad(x, padding):
    if x.ndim == 3:
        top_pad = x[0] * np.ones((padding[0], 1, 1))
        pad_args = (0, 0), (padding[1], padding[1]), (0, 0)
    else:
        top_pad = x[0] * np.ones((padding[0], 1))
        pad_args = (0, 0), (padding[1], padding[1])
    padded = np.concatenate((top_pad, x, np.roll(x[-1:-padding[0] - 1:-1], 180, axis=1)), axis=0)
    return np.pad(padded, pad_args, mode='wrap')


def extract_patches(x, patch_size):
    pad_size = (patch_size - 1) // 2
    if x.ndim == 3:
        top_pad = x[0] * np.ones((pad_size, 1, 1))
        pad_args = (0, 0), (pad_size, pad_size), (0, 0)
        patch_shape = (patch_size, patch_size, x.shape[-1])
    else:
        top_pad = x[0] * np.ones((pad_size, 1))
        pad_args = (0, 0), (pad_size, pad_size)
        patch_shape = (patch_size, patch_size)
    padded = np.concatenate((top_pad, x, np.roll(x[-1:-pad_size - 1:-1], 180, axis=1)), axis=0)
    padded = np.pad(padded, pad_args, mode='wrap')
    patches = view_as_windows(padded, patch_shape)
    return patches.reshape(x.shape[:2] + (-1,))


def estimate_background(x, patch_size):
    if not patch_size % 2:
        patch_size += 1
    patches = extract_patches(x, patch_size)
    return bn.nanmean(patches, axis=-1)


def get_tec_map_interval(year, month, index, time_radius=2):
    day = index // 24 + 1
    hour = index % 24
    tec_time = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00")
    start_time = tec_time - np.timedelta64(time_radius, 'h')
    start_ut = utils.datetime64_to_timestamp(start_time)
    start_year, start_month = get_datetime64_year_month(start_time)
    end_time = tec_time + np.timedelta64(time_radius + 1, 'h')
    end_ut = utils.datetime64_to_timestamp(end_time)
    end_year, end_month = get_datetime64_year_month(end_time)
    tec, _, ut = get_tec_maps(start_year, start_month)
    if (start_year, start_month) != (end_year, end_month):
        tec2, _, ut2 = get_tec_maps(end_year, end_month)
        ut = np.concatenate((ut, ut2))
        tec = np.concatenate((tec, tec2), axis=-1)
    sl = slice(np.argmax(ut >= start_ut), np.argmax(ut > end_ut))
    return ut[sl], tec[:, :, sl]


def preprocess_tec_interval(tec, ma_width=19):
    # get the middle tec map
    idx = tec.shape[-1] // 2
    # throw away outlier values
    mask = np.isfinite(tec)
    mask[mask] = tec[mask] > 150
    tec[mask] = np.nan
    # change to log
    logtec = np.log10(tec + .001)
    # estimate background
    bg = estimate_background(logtec, ma_width)
    # subtract background
    return logtec[:, :, idx] - bg


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


def downsample_2d_array(arr, ds):
    windows = view_as_windows(arr, ds, ds)
    return bn.nanmean(windows.reshape(windows.shape[:2] + (-1, )), axis=-1)


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


class RbfIversion:

    def __init__(self, mlt_grid, mlat_grid, ds=None, **params):
        self.params = DEFAULT_PARAMS.copy()
        self.params.update(**params)
        self.ds = ds
        if ds is not None:
            self.mlt_grid = downsample_2d_array(mlt_grid, ds)
            self.mlat_grid = downsample_2d_array(mlat_grid, ds)
            self.output_mlt = mlt_grid[0]
            self.output_mlat = mlat_grid[:, 0]
        else:
            self.mlt_grid = mlt_grid
            self.mlat_grid = mlat_grid
            self.output_mlt = None
            self.output_mlat = None

        self.d = self.mlt_grid.size
        self.shape = self.mlt_grid.shape
        mlat = np.deg2rad(self.mlat_grid[:, 0])
        if self.params['cos_mlat']:
            mlat_cost_weight = np.cos(mlat)
        else:
            mlat_cost_weight = np.ones_like(mlat)
        self.mlat_cost_weight = (mlat_cost_weight[:, None] * np.ones((1, self.shape[1]))).ravel()

        self.basis = get_rbf_matrix(self.shape, self.params['rbf_bw'])
        self.tv = get_tv_matrix(self.shape, hw=self.params['tv_hw'], vw=self.params['tv_vw'])

        self.tv_weight = cp.Parameter(nonneg=True)
        self.tv_weight.value = self.params['tv_weight']
        self.l2_weight = cp.Parameter(nonneg=True)
        self.l2_weight.value = self.params['l2_weight']

    def run(self, x, ut):
        model = trough_model.get_model(ut, self.mlt_grid[0])
        model_weight = (self.mlat_grid - model[None, :]) ** 2
        model_weight = model_weight.ravel()
        model_weight /= (model_weight.max() / self.params['model_weight_max'])
        model_weight += 1

        fin_mask = np.isfinite(x.ravel())
        finx = x.ravel()[fin_mask]
        u = cp.Variable(self.d)
        main_cost = u.T @ self.basis[fin_mask, :].T @ cp.multiply(self.mlat_cost_weight[fin_mask], finx)
        tv_cost = self.tv_weight * cp.norm1(self.tv @ u)
        l2_cost = self.l2_weight * model_weight @ (u ** 2)
        total_cost = main_cost + tv_cost + l2_cost
        prob = cp.Problem(cp.Minimize(total_cost), [u >= 0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            prob.solve(solver=cp.SCS, max_iters=2000)
        return u.value

    @staticmethod
    def decision(u):
        return u >= 1

    def postprocess_labels(self, u, wrap=10):
        if u.sum() == 0:
            return u
        padded = np.pad(u, ((0, 0), (wrap, wrap)), mode='wrap')
        labeled = measure.label(padded, connectivity=2)
        props = pandas.DataFrame(measure.regionprops_table(labeled, properties=('label', 'area', 'perimeter')))
        labeled = labeled[:, wrap:-wrap]
        for i, r in props[props['perimeter'] < self.params['perimeter_th']].iterrows():
            u[labeled == r['label']] = 0
        return u

    def postprocess(self, u):
        f = interp2d(self.mlt_grid[0], self.mlat_grid[:, 0], u.reshape(self.shape))
        return f(self.output_mlt, self.output_mlat)

    def load_and_preprocess(self, year, month, index):
        ut, tec = get_tec_map_interval(year, month, index, time_radius=self.params['bge_temporal_rad'])
        if ut.shape[0] < (2 * self.params['bge_temporal_rad'] + 1):
            return None, None, None
        ut = ut[ut.shape[0] // 2]
        original = tec[:, :, tec.shape[-1] // 2]
        x = preprocess_tec_interval(tec, self.params['bge_spatial_size'])
        return downsample_2d_array(x, self.ds), ut, original
