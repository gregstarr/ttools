import pandas as pd
import numpy as np
import bottleneck as bn
from skimage.util import view_as_windows, pad

from ttools.trough_model import get_model


def normalize_features(X):
    m = bn.nanmean(X, axis=0)
    s = bn.nanstd(X, axis=0)
    return (X - m) / s


def get_features(tec, mlt, mlat, ut, patch_size=3):
    """features: mlat, mlt, tec, gradients, distance above model, distance below model, distance below auroral oval,
    collect all these features for a 3x3 or 5x5 area around a particular pixel
    """
    # already have tec, mlat
    # gradient
    grad = np.gradient(tec, axis=0)
    grad1 = np.maximum(0, grad)
    grad2 = np.maximum(0, -grad)
    # stack together and extract patches
    image = np.stack((tec, grad1, grad2), axis=-1)
    pad_size = (patch_size - 1) // 2
    patches = view_as_windows(pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='wrap'),
                              (patch_size, patch_size, image.shape[-1]))
    patches = patches.reshape(image.shape[:2] + (-1,))  # turn back to many-channeled image
    patches = patches.reshape((-1, patches.shape[2]))  # turn to list of feature vectors
    # turn mlt into sin and cos
    theta = mlt * np.pi / 12
    sin_mlt = np.sin(theta)
    cos_mlt = np.cos(theta)
    # get the modelled mlat
    model = get_model(ut, mlt[0])
    # create features for above and below model
    above = np.maximum(0, mlat - model)
    below = np.maximum(0, model - mlat)
    # add in other features
    features = np.column_stack((patches, sin_mlt.ravel(), cos_mlt.ravel(), above.ravel(), below.ravel()))
    return features


def get_lda_params(X, y):
    n1 = np.sum(y == 0)
    n2 = np.sum(y == 1)
    m1 = np.nanmean(X[y == 0], axis=0)
    m2 = np.nanmean(X[y == 1], axis=0)
    df1 = pd.DataFrame(X[y == 0])
    df2 = pd.DataFrame(X[y == 1])
    S = (n1 * df1.cov() + n2 * df2.cov()) / (n1 + n2)
    reg = 0
    w = np.linalg.inv(S + np.eye(S.shape[0]) * reg) @ (m2 - m1)
    return w


def get_pca_params(X, k=2):
    df = pd.DataFrame(X)
    evals, evecs = np.linalg.eigh(df.cov())
    print(f"variance explained: {evals[:-k-1:-1].sum() / evals.sum()}")
    return evals[:-k-1:-1], evecs[:, :-k-1:-1]


def pca_projection(X, w):
    m = bn.nanmean(X, axis=0)
    return np.where(np.isfinite(X), X - m, 0) @ w
