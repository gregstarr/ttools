import numpy as np
import bottleneck as bn
import pandas
from skimage import measure, morphology

from ttools import utils, config


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
    patches = utils.extract_patches(tec, patch_shape)
    return bn.nanmean(patches.reshape((tec.shape[0] - patch_shape[0] + 1,) + tec.shape[1:] + (-1,)), axis=-1)


def preprocess_interval(tec, times, min_val=0, max_val=100, bg_est_shape=(3, 15, 15)):
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
    return x, t


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


def remove_auroral(inp, arb):
    output = inp.copy()
    output *= (config.mlat_grid[None, :, :] < arb[:, None, :])
    return output


def postprocess(initial_trough, perimeter_th=50, area_th=1, arb=None, closing_r=0):
    trough = initial_trough.copy()
    if closing_r > 0:
        selem = morphology.disk(closing_r, dtype=bool)[:, :, None]
        trough = np.pad(trough, ((0, 0), (0, 0), (closing_r, closing_r)), 'wrap')
        trough = morphology.binary_closing(trough, selem)[:, :, closing_r:-closing_r]
    if arb is not None:
        trough = remove_auroral(trough, arb)
    for t in range(trough.shape[0]):
        tmap = trough[t]
        labeled = measure.label(tmap, connectivity=2)
        labeled = fix_boundaries(labeled)
        props = pandas.DataFrame(measure.regionprops_table(labeled, properties=('label', 'area', 'perimeter')))
        error_mask = (props['perimeter'] < perimeter_th) | (props['area'] < area_th)
        for i, r in props[error_mask].iterrows():
            tmap[labeled == r['label']] = 0
        trough[t] = tmap
    return trough
