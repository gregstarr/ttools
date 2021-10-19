import numpy as np
from skimage import measure, draw

from ttools import tec as ttec, utils, config
from ttools.trough_labeling.labeler import TroughLabelJob


class ConstantBaselineLabelJob(TroughLabelJob):

    def __init__(self, date, bg_est_shape=(1, 1, 1)):
        super().__init__(date, bg_est_shape=bg_est_shape)

    def run(self):
        mlat = np.broadcast_to(config.mlat_grid[None], self.x.shape)
        self.trough = abs(mlat - 65) <= 2

    @staticmethod
    def get_random_params():
        ...

    def plot(self, swarm_troughs, plot_dir):
        ...
