import abc
import numpy as np
import os

from ttools import tec as ttec, io, utils


class TroughLabelJobManager:

    def __init__(self, job_class, **kwargs):
        self.job_class = job_class
        self._job = job_class(None, **kwargs)
        self._kwargs = kwargs
        self.save_dir = None

    @classmethod
    def get_random(cls, job_class, params_dir):
        params = job_class.get_random_params()
        io.write_yaml(os.path.join(params_dir, 'params.yaml'), **params)
        obj = cls(job_class, **params)
        obj.save_dir = params_dir
        return obj

    def make_job(self, date):
        return self.job_class(date, **self._kwargs)

    def __getattr__(self, item):
        return getattr(self._job, item)


class TroughLabelJob(abc.ABC):

    def __init__(self, date, **kwargs):
        self.date = date
        self.bg_est_shape = kwargs['bg_est_shape']
        self.tec = None
        self.tec_times = None
        self.x = None
        self.times = None
        self.ssmlon = None
        self.arb = None
        if date is not None:
            self.load_data()
        self.model_output = None
        self.trough = None

    def load_data(self):
        one_h = np.timedelta64(1, 'h')
        start_time = self.date.astype('datetime64[D]').astype('datetime64[s]')
        end_time = start_time + np.timedelta64(1, 'D')
        tec_start = start_time - np.floor(self.bg_est_shape[0] / 2) * one_h
        tec_end = end_time + (np.floor(self.bg_est_shape[0] / 2)) * one_h

        self.tec, self.tec_times, ssmlon, n = io.get_tec_data(tec_start, tec_end)
        self.x, self.times = ttec.preprocess_interval(self.tec, self.tec_times, bg_est_shape=self.bg_est_shape)
        self.ssmlon, = utils.moving_func_trim(self.bg_est_shape[0], ssmlon)
        self.arb, _ = io.get_arb_data(start_time, end_time)

    @abc.abstractmethod
    def run(self):
        ...

    @staticmethod
    @abc.abstractmethod
    def get_random_params():
        ...

    @abc.abstractmethod
    def plot(self, swarm_troughs, plot_dir):
        ...
