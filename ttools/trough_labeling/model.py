import matplotlib.pyplot as plt
import os

from ttools import tec as ttec, utils, config, deminov, plotting
from ttools.trough_labeling.labeler import TroughLabelJob


class ModelBaselineLabelJob(TroughLabelJob):

    def __init__(self, date, bg_est_shape=(1, 1, 1)):
        super().__init__(date, bg_est_shape=bg_est_shape)

    def run(self):
        ut = self.times.astype('datetime64[s]').astype(float)
        model_mlat = deminov.get_model(ut, config.mlt_vals)
        self.trough = abs(config.mlat_grid[None, :, :] - (model_mlat[:, None, :] + 1)) <= 3

    @staticmethod
    def get_random_params():
        ...

    def _plot_single(self, i, swarm_troughs, plot_dir):
        plots = {
            'model_trough': {'data': self.trough[i], 'kwargs': dict(cmap='Blues')},
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
