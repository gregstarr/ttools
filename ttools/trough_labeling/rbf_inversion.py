"""
steps:
    - get data interval
    - preprocess
        - log, background estimation
    - setup and run optimization problem
    - threshold

Example:
```
# load data
tec, times, ssmlon, n = io.get_tec_data(start_time, end_time)
# preprocess
x, times = ttec.preprocess_interval(tec, times)
# setup optimization
args = rbf_inversion.get_optimization_args(x, times)
# run optimization
model_output = rbf_inversion.run_multiple(args)
# threshold
initial_trough = model_output >= 1
# postprocess
trough = ttec.postprocess(initial_trough)
```
"""
import numpy as np
import cvxpy as cp
import multiprocessing
from scipy import stats
import matplotlib.pyplot as plt
import os

from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csr_matrix

from ttools import tec as ttec, deminov, config, io, utils, plotting
from ttools.trough_labeling.labeler import TroughLabelJob


def get_rbf_matrix(shape, bandwidth=1):
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xy = np.column_stack((X.ravel(), Y.ravel()))
    gamma = np.log(2) / bandwidth ** 2
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


def get_optimization_args(x, times, model_mlat, model_weight_max, rbf_bw, tv_hw, tv_vw, l2_weight, tv_weight,
                          prior_order, mlat_grid=None):
    if mlat_grid is None:
        mlat_grid = config.mlat_grid
    all_args = []
    # get rbf basis matrix
    basis = get_rbf_matrix(x.shape[1:], rbf_bw)
    # get tv matrix
    tv = get_tv_matrix(x.shape[1:], tv_hw, tv_vw) * tv_weight

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
    if x.size == 0:
        return np.zeros(output_shape)
    main_cost = u.T @ basis.T @ x
    tv_cost = cp.norm1(tv @ u)
    l2_cost = l2 @ (u ** 2)
    total_cost = main_cost + tv_cost + l2_cost
    prob = cp.Problem(cp.Minimize(total_cost))
    try:
        prob.solve(solver=config.SOLVER)
    except Exception as e:
        print("FAILED, USING ECOS:", e)
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


PARAM_SAMPLING = {
    'tv_weight': stats.loguniform(.001, 1),
    'l2_weight': stats.loguniform(.001, 1),
    'bge_spatial_size': stats.randint(4, 12),
    'bge_temporal_rad': stats.randint(0, 3),
    'rbf_bw': stats.randint(1, 4),
    'tv_hw': stats.randint(1, 4),
    'tv_vw': stats.randint(1, 4),
    'model_weight_max': stats.randint(1, 20),
    'perimeter_th': stats.randint(10, 100),
    'area_th': stats.randint(10, 100),
    'prior_order': [1, 2],
    'prior': ['empirical_model', 'auroral_boundary'],
    'prior_offset': stats.randint(-5, 0),
}

BG_EST_SHAPE = (1, 19, 15)
MODEL_WEIGHT_MAX = 15
RBF_BW = 1
TV_HW = 2
TV_VW = 1
L2_WEIGHT = .07
TV_WEIGHT = .15
PERIMETER_TH = 40
AREA_TH = 40
PRIOR_ORDER = 1
PRIOR = 'auroral_boundary'
PRIOR_OFFSET = -3
THRESHOLD = 1
CLOSING_RAD = 0


class RbfInversionLabelJob(TroughLabelJob):

    def __init__(self, date, bg_est_shape=BG_EST_SHAPE, model_weight_max=MODEL_WEIGHT_MAX, rbf_bw=RBF_BW, tv_hw=TV_HW,
                 tv_vw=TV_VW, l2_weight=L2_WEIGHT, tv_weight=TV_WEIGHT, prior_order=PRIOR_ORDER, prior=PRIOR,
                 prior_offset=PRIOR_OFFSET, perimeter_th=PERIMETER_TH, area_th=AREA_TH, closing_rad=0,
                 threshold=THRESHOLD):
        super().__init__(date, bg_est_shape, model_weight_max, rbf_bw, tv_hw, tv_vw, l2_weight, tv_weight, prior_order,
                         prior, prior_offset, perimeter_th, area_th, threshold)
        self.model_weight_max = model_weight_max
        self.rbf_bw = rbf_bw
        self.tv_vw = tv_vw
        self.tv_hw = tv_hw
        self.l2_weight = l2_weight
        self.tv_weight = tv_weight
        self.prior_order = prior_order
        self.prior = prior
        self.prior_offset = prior_offset
        self.perimeter_th = perimeter_th
        self.area_th = area_th
        self.threshold = threshold
        self.closing_rad = closing_rad

    def run(self):
        print("Setting up inversion optimization")
        if self.prior == 'empirical_model':
            # get deminov model
            ut = self.times.astype('datetime64[s]').astype(float)
            model_mlat = deminov.get_model(ut, config.mlt_vals)
        elif self.prior == 'auroral_boundary':
            model_mlat = self.arb + self.prior_offset
        else:
            raise Exception("Invalid prior name")

        args = get_optimization_args(self.x, self.times, model_mlat, self.model_weight_max, self.rbf_bw, self.tv_hw,
                                     self.tv_vw, self.l2_weight, self.tv_weight, self.prior_order)
        # run optimization
        print("Running inversion optimization")
        self.model_output = run_multiple(args, parallel=config.PARALLEL)
        if self.threshold is not None:
            # threshold
            initial_trough = self.model_output >= self.threshold
            # postprocess
            print("Postprocessing inversion results")
            self.trough = ttec.postprocess(initial_trough, self.perimeter_th, self.area_th, self.arb, self.closing_rad)

    @staticmethod
    def get_random_params():
        params = {}
        bge_temporal_size = 0
        bge_spatial_size = 0
        for p in PARAM_SAMPLING:
            if isinstance(PARAM_SAMPLING[p], list):
                val = PARAM_SAMPLING[p][np.random.randint(len(PARAM_SAMPLING[p]))]
            else:
                try:
                    val = PARAM_SAMPLING[p].rvs().item()
                except:
                    val = PARAM_SAMPLING[p].rvs()
            if p == 'bge_temporal_rad':
                bge_temporal_size = val * 2 + 1
            elif p == 'bge_spatial_size':
                bge_spatial_size = val * 2 + 1
            else:
                params[p] = val
        params['bg_est_shape'] = (bge_temporal_size, bge_spatial_size, bge_spatial_size)
        params['threshold'] = None
        return params

    def _plot_single(self, i, swarm_troughs, plot_dir):
        plots = {
            'tec': {'data': self.tec[i + self.bg_est_shape[0] // 2], 'kwargs': dict(vmin=0, vmax=16)},
            'x': {'data': self.x[i], 'kwargs': dict(vmin=-.5, vmax=.5, cmap='coolwarm')},
            'output': {'data': self.model_output[i], 'kwargs': dict(vmin=-1.5, vmax=1.5, cmap='coolwarm')},
            'trough': {'data': self.trough[i], 'kwargs': dict(cmap='Blues')},
        }
        for name, plot in plots.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'), tight_layout=True)
            pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, plot['data'], **plot['kwargs'])
            plotting.plot_swarm_troughs_polar(ax, swarm_troughs)
            plotting.plot_mlon_lines(ax, self.ssmlon[i])
            plotting.plot_arb(ax, config.mlt_vals, self.arb[i])
            plotting.format_polar_mag_ax(ax)
            ax.set_title(f"{self.times[i]} {name}")
            plt.colorbar(pcm)
            fig.savefig(os.path.join(plot_dir, f"{self.date.astype('datetime64[D]')}_{i}_{name}.png"))
            plt.close(fig)

    def plot(self, swarm_troughs, plot_dir):
        for i in range(self.times.shape[0]):
            st = swarm_troughs[i == (swarm_troughs['tec_ind'] % 24)]
            self._plot_single(i, st, plot_dir)
