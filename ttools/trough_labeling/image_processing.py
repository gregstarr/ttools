import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt

from ttools import tec as ttec, config, utils, plotting
from ttools.trough_labeling.labeler import TroughLabelJob


def get_sparse_diff(x, axis, size, pad_wrap, median_shape=None, softmax_multiplier=0, mean_shape=None):
    pad_arg = [(0, 0)] * 3
    pad_arg[axis] = (1, 0)
    pad = np.pad(x, pad_arg, pad_wrap)
    grad = np.diff(pad, axis=axis)
    grad[np.isnan(grad)] = 0
    if median_shape is not None:
        grad = np.median(utils.extract_patches(grad, median_shape), axis=(-1, -2, -3))
    max_filter_shape = [1, 1, 1]
    max_filter_shape[axis] = size
    patches = utils.extract_patches(abs(grad), max_filter_shape)
    slicer = [slice(None), slice(None), slice(None), 0, 0, 0]
    slicer[3 + axis] = size // 2
    slicer = tuple(slicer)
    if softmax_multiplier != 0:
        patches *= softmax_multiplier
        exp = np.exp(patches - np.nanmax(patches, axis=(-1, -2, -3), keepdims=True))
        exp = exp / np.nansum(exp, axis=(-1, -2, -3), keepdims=True)
        max_id = exp[slicer]
        grad = np.nansum(utils.extract_patches(grad, max_filter_shape), axis=(-1, -2, -3)) * max_id
    else:
        max_id = (patches == np.nanmax(patches, axis=(-1, -2, -3), keepdims=True))
        max_id = max_id[slicer]
        grad = np.nansum(utils.extract_patches(grad, max_filter_shape), axis=(-1, -2, -3)) * max_id
    if mean_shape is not None:
        grad = np.mean(utils.extract_patches(grad, mean_shape), axis=(-1, -2, -3))
    return grad


PARAM_SAMPLING = {
    'bge_spatial_size': stats.randint(4, 12),
    'bge_temporal_rad': stats.randint(0, 3),
    'prior_weight': stats.uniform(0, 1.5),
    'tv_weight': stats.uniform(0, 1.5),
    'perimeter_th': stats.randint(10, 100),
    'area_th': stats.randint(10, 100),
    'prior_offset': stats.randint(-5, 0),
    'prior_order': [1, 2],
    'x_max_filter_size': stats.randint(0, 6),
    'y_max_filter_size': stats.randint(0, 6),
    'mean_filter_size': stats.randint(-1, 6),
    'median_filter_size': stats.randint(-1, 6),
    'softmax_scale': stats.randint(10, 200),
    'softmax_enable': [0, 1],
    'closing_rad': [0, 1, 2, 3],
}


BG_EST_SHAPE = (1, 19, 11)
PRIOR_WEIGHT = .5
TV_WEIGHT = .3
PERIMETER_TH = 40
AREA_TH = 40
PRIOR_OFFSET = -3
PRIOR_ORDER = 1
X_MAX_FILTER_SIZE = 5
Y_MAX_FILTER_SIZE = 5
MEAN_FILTER_SIZE = 5
MEDIAN_FILTER_SIZE = 5
SOFTMAX_SCALE = 0
THRESHOLD = 0.1
CLOSING_RAD = 0


class ImageProcessingLabelJob(TroughLabelJob):

    def __init__(self, date, bg_est_shape=BG_EST_SHAPE, prior_weight=PRIOR_WEIGHT, tv_weight=TV_WEIGHT,
                 perimeter_th=PERIMETER_TH, area_th=AREA_TH, prior_offset=PRIOR_OFFSET, prior_order=PRIOR_ORDER,
                 x_max_filter_size=X_MAX_FILTER_SIZE, y_max_filter_size=Y_MAX_FILTER_SIZE,
                 mean_filter_size=MEAN_FILTER_SIZE, median_filter_size=MEDIAN_FILTER_SIZE, softmax_scale=SOFTMAX_SCALE,
                 threshold=THRESHOLD, closing_rad=CLOSING_RAD):
        super().__init__(date, bg_est_shape, prior_weight, tv_weight, perimeter_th, area_th, prior_offset, prior_order,
                         x_max_filter_size, y_max_filter_size, mean_filter_size, median_filter_size, softmax_scale,
                         threshold, closing_rad)
        self.prior_weight = prior_weight
        self.tv_weight = tv_weight
        self.perimeter_th = perimeter_th
        self.area_th = area_th
        self.prior_offset = prior_offset
        self.prior_order = prior_order
        self.x_max_filter_size = x_max_filter_size
        self.y_max_filter_size = y_max_filter_size
        self.mean_filter_size = mean_filter_size
        self.median_filter_size = median_filter_size
        self.softmax_scale = softmax_scale
        self.closing_rad = closing_rad
        self.threshold = threshold

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
        params['softmax_scale'] = params['softmax_scale'] * params['softmax_enable']
        del params['softmax_enable']
        params['x_max_filter_size'] = 2 * params['x_max_filter_size'] + 1
        params['y_max_filter_size'] = 2 * params['y_max_filter_size'] + 1
        params['mean_filter_size'] = 2 * params['mean_filter_size'] + 1
        params['median_filter_size'] = 2 * params['median_filter_size'] + 1
        params['threshold'] = None
        return params

    def run(self):
        prior_mlat = self.arb + self.prior_offset
        self.prior = abs(config.mlat_grid[None, :, :] - prior_mlat[:, None, :]) ** self.prior_order
        self.prior -= np.min(self.prior, axis=1, keepdims=True)
        self.prior /= np.max(self.prior)
        self.prior *= self.prior_weight

        median_filter_shape = None
        if self.median_filter_size > 0:
            median_filter_shape = (1, self.median_filter_size, self.median_filter_size)

        mean_filter_shape_x = None
        mean_filter_shape_y = None
        if self.mean_filter_size > 0:
            mean_filter_shape_y = (1, 1, self.mean_filter_size)
            mean_filter_shape_x = (1, self.mean_filter_size, 1)

        self.gy = get_sparse_diff(self.x, 1, self.y_max_filter_size, 'edge', median_filter_shape, self.softmax_scale,
                                  mean_filter_shape_y)
        self.gx = get_sparse_diff(self.x, 2, self.x_max_filter_size, 'wrap', median_filter_shape, self.softmax_scale,
                                  mean_filter_shape_x)

        self.tv = np.cumsum(self.gx, axis=2) + np.cumsum(-1 * self.gx[:, :, ::-1], axis=2)[:, :, ::-1]
        self.tv += np.cumsum(self.gy, axis=1) + np.cumsum(-1 * self.gy[:, ::-1, :], axis=1)[:, ::-1, :]
        self.tv /= 2

        self.model_output = -1 * (self.x + self.prior + self.tv_weight * self.tv)
        if self.threshold is not None:
            # threshold
            initial_trough = self.model_output >= self.threshold
            # postprocess
            print("Postprocessing inversion results")
            self.trough = ttec.postprocess(initial_trough, self.perimeter_th, self.area_th, self.arb, self.closing_rad)

    def _plot_single(self, i, swarm_troughs, plot_dir):
        plots = {
            'tec': {'data': self.tec[i + self.bg_est_shape[0] // 2], 'kwargs': dict(vmin=0, vmax=16)},
            'x': {'data': self.x[i], 'kwargs': dict(vmin=-1, vmax=1, cmap='coolwarm')},
            'prior': {'data': self.prior[i], 'kwargs': dict(vmin=-1, vmax=1, cmap='coolwarm')},
            'g_x': {'data': self.gx[i], 'kwargs': dict(vmin=-.25, vmax=.25, cmap='coolwarm')},
            'g_y': {'data': self.gy[i], 'kwargs': dict(vmin=-.25, vmax=.25, cmap='coolwarm')},
            'tv': {'data': self.tv[i], 'kwargs': dict(vmin=-1, vmax=1, cmap='coolwarm')},
            'output': {'data': self.model_output[i], 'kwargs': dict(vmin=-1, vmax=1, cmap='coolwarm')},
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
