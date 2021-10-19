import numpy as np
from skimage import measure, draw
import matplotlib.pyplot as plt
import os

from ttools import tec as ttec, utils, plotting, config
from ttools.trough_labeling.labeler import TroughLabelJob


BG_EST_SHAPE = (1, 19, 15)
MEDIAN_SHAPE = (1, 5, 5)
PERIMETER_TH = 40
AREA_TH = 40
THRESHOLD = -.2


class AaBaselineLabelJob(TroughLabelJob):

    def __init__(self, date, bg_est_shape=BG_EST_SHAPE, perimeter_th=PERIMETER_TH, area_th=AREA_TH,
                 median_shape=MEDIAN_SHAPE, threshold=THRESHOLD):
        super().__init__(date, bg_est_shape=bg_est_shape)
        self.perimeter_th = perimeter_th
        self.area_th = area_th
        self.median_shape = median_shape
        self.threshold = threshold

    def run(self):
        xp = np.nanmedian(utils.extract_patches(self.x, self.median_shape), axis=(-1, -2, -3))
        xp[np.isnan(self.x)] = 0
        initial_trough = np.zeros_like(self.x, dtype=bool)
        for t in range(self.x.shape[0]):
            contours = measure.find_contours(xp[t], -.05)
            for contour in contours:
                if np.all(contour[0] == contour[-1]):
                    rr, cc = draw.polygon(contour[:, 0], contour[:, 1], self.x[0].shape)
                    if np.any(self.x[t, rr, cc] < self.threshold):
                        initial_trough[t, rr, cc] = 1
        self.trough = ttec.postprocess(initial_trough, self.perimeter_th, self.area_th, self.arb)

    @staticmethod
    def get_random_params():
        ...

    def _plot_single(self, i, swarm_troughs, plot_dir):
        plots = {
            'aa_trough': {'data': self.trough[i], 'kwargs': dict(cmap='Blues')},
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
