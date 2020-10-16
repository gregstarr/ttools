import numpy as np
import apexpy
from scipy.interpolate import interp1d
import datetime

from ttools.io import get_gm_index


def get_model(ut, mlt):
    kp = get_weighted_kp(ut)
    converter = apexpy.Apex(date=datetime.datetime.fromtimestamp(ut))
    lat = 65.5 * np.ones_like(mlt)
    for i in range(10):
        lon = _model_subroutine_lon(mlt, lat, ut, converter)
        lat = _model_subroutine_lat(mlt, lon, kp)
    return lat


def _model_subroutine_lon(mlt, mlat, ut, converter):
    """

    Parameters
    ----------
    mlt: numpy.ndarray (n_mlt, )
    mlat: numpy.ndarray (n_mlt, )
    ut: int
    converter: apexpy.Apex

    Returns
    -------
    glon: numpy.ndarray (n_mlt, )
    """
    lat, lon = converter.convert(mlat, mlt, 'mlt', 'geo', 350, datetime=datetime.datetime.fromtimestamp(ut))
    return lon


def _model_subroutine_lat(mlt, glon, kp):
    """

    Parameters
    ----------
    mlt: numpy.ndarray (n_mlt, )
    glon: numpy.ndarray (n_mlt, )
    kp: float

    Returns
    -------
    mlat: numpy.ndarray (n_t, n_mlt)
    """
    phi_t = 3.16 - 5.6 * np.cos(np.deg2rad(15 * (mlt - 2.4))) + 1.4 * np.cos(np.deg2rad(15 * (2 * mlt - .8)))
    phi_lon = .85 * np.cos(np.deg2rad(glon + 63)) - .52 * np.cos(np.deg2rad(2 * glon + 5))
    return 65.5 - 2.4 * kp + phi_t + phi_lon * np.exp(-.3 * kp)


def get_weighted_kp(ut, fn="E:\\2000_2020_kp_ap.txt", tau=.6, T=10):
    times, ap = get_gm_index(fn)
    prehistory = np.column_stack([ap[T - i - 1:ap.shape[0] - i] for i in range(T)])
    weight_factors = tau ** np.arange(T)
    ap_tau = np.sum((1 - tau) * prehistory * weight_factors, axis=1)
    kp_tau = 2.1 * np.log(.2 * ap_tau + 1)
    times = times[T - 1:]
    kp = interp1d(times, kp_tau, kind='previous')
    return kp(ut)
