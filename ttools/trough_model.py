import numpy as np
import apexpy
from scipy.interpolate import interp1d
import datetime

from ttools.io import get_gm_index_kyoto


def get_model(ut, mlt):
    """Get magnetic latitudes of the trough according to the model in Deminov 2017
    for a specific time and set of magnetic local times.

    Parameters
    ----------
    ut: int
        unix timestamp
    mlt: numpy.ndarray (n, )
        magnetic local times to evaluate model at
    Returns
    -------
    mlat: numpy.ndarray (n, )
        model evaluated at the given magnetic local times
    """
    kp = _get_weighted_kp(ut)
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


def _get_weighted_kp(ut, fn="E:\\2000_2020_kp_ap.txt", tau=.6, T=10):
    """Get a weighed sum of kp values over time. See paper for details.

    Parameters
    ----------
    ut: int
        unix timestamp to get the weighted kp for
    fn: str
        file name with kp and ap downloaded from http://wdc.kugi.kyoto-u.ac.jp/kp/index.html#LIST
    tau: float
        decay factor to weight previous time steps with
    T: int
        number of previous time steps to include

    Returns
    -------
    weighted kp: float
    """
    df = get_gm_index_kyoto(fn)
    ap = df['ap'].values
    times = np.array(df['ap'].index.values.astype(float) / 1e9, dtype=int)
    prehistory = np.column_stack([ap[T - i - 1:ap.shape[0] - i] for i in range(T)])
    weight_factors = tau ** np.arange(T)
    ap_tau = np.sum((1 - tau) * prehistory * weight_factors, axis=1)
    kp_tau = 2.1 * np.log(.2 * ap_tau + 1)
    times = times[T - 1:]
    kp = interp1d(times, kp_tau, kind='previous')
    return kp(ut)
