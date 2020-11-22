import pandas as pd
import numpy as np
import bottleneck as bn
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import haversine_distances


def get_spherical_rbf_matrix(lat_vals, lon_vals, bandwidth, downsampling=2):
    imw = lon_vals.shape[0]
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    latlon_ds = np.column_stack((lat_vals[::downsampling], lon_grid[::downsampling, 0]))
    latlon_full = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    osize = latlon_full.shape[0]
    dsize = osize // downsampling**2
    dist = haversine_distances(latlon_full, latlon_ds)
    scaled_bandwidth = np.where(dist == 0, 999, dist).min(axis=0).max() * bandwidth
    basis = np.exp(-(dist / scaled_bandwidth) ** 2)
    basis[basis < .01] = 0

    cols = downsampling * np.arange(dsize)[None, :] // imw
    rows = (imw * (np.arange(osize)[:, None] // imw) + (np.arange(osize)[:, None] - np.arange(dsize)[None, :] * 2) % imw)

    basis = basis[rows, cols]
    return csr_matrix(basis)


def normalize_features(X):
    m = bn.nanmean(X, axis=0)
    s = bn.nanstd(X, axis=0)
    return (X - m) / s


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
