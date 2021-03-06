import numpy as np

from ttools import rbf_inversion, config, io


def test_preprocess_interval():
    T = 10
    nlat = 100
    nlon = 100
    bg_est_shape = (3, 21, 21)
    times = np.datetime64("2000") + np.arange(T) * np.timedelta64(1, 'h')
    background = np.sin(np.linspace(0, 2 * np.pi, nlon))[None, None, :] * np.ones((T, nlat, nlon))
    signal = np.sin(10 * np.linspace(0, 2 * np.pi, nlon))[None, None, :] * np.ones((T, nlat, nlon))
    log_tec = background + signal
    tec = 10 ** log_tec
    tec[1, 20, 20] = 200
    tec[2, 40, 40] = -5
    det_log_tec, trim_time = rbf_inversion.preprocess_interval(tec, times, bg_est_shape=bg_est_shape)
    assert np.nanmean(abs(det_log_tec - signal[1:-1])) < .1
    assert trim_time.shape[0] == T - bg_est_shape[0] + 1
    assert det_log_tec.shape == (T - bg_est_shape[0] + 1, nlat, nlon)


def test_model_artificial_example():
    times = np.datetime64("2000") + np.arange(4) * np.timedelta64(1, 'h')
    # nominal trough
    trough1 = abs(config.mlat_grid - 65) <= 2
    # partial trough
    trough2 = (abs(config.mlat_grid - 65) <= 2) * (abs(config.mlt_grid) < 3)
    # high trough - reject
    trough3 = (config.mlat_grid > 83) * (abs(config.mlt_grid) < 3)
    # low trough - reject
    trough4 = (config.mlat_grid < 35) * (abs(config.mlt_grid) < 2)
    # convert to det log tec
    basis = rbf_inversion.get_rbf_matrix(config.mlat_grid.shape)
    det_log_tec = -.2 * (basis @ np.column_stack((trough1.ravel(), trough2.ravel(), trough3.ravel(), trough4.ravel()))).T
    shp = (4, ) + config.mlat_grid.shape
    det_log_tec = det_log_tec.reshape(shp)
    det_log_tec += np.random.randn(*shp) * .1
    args = rbf_inversion.get_optimization_args(det_log_tec, times, model_weight_max=30, l2_weight=.15, tv_weight=.06)
    output = np.empty_like(det_log_tec)
    for i, a in enumerate(args):
        output[i] = rbf_inversion.run_single(*a)
    print(output[0][trough1].mean(), output[1][trough2].mean(), output[2][trough3].mean(), output[3][trough4].mean())
    assert output[0][trough1].mean() > 1
    assert output[1][trough2].mean() > 1
    assert output[2][trough3].mean() < 1
    assert output[3][trough4].mean() < 1


def test_postprocess():
    """verify that small troughs are rejected, verify that troughs that wrap around the border are not incorrectly
    rejected
    """
    good_trough = (abs(config.mlat_grid - 65) < 3) * (abs(config.mlt_grid) < 2)
    small_reject = (abs(config.mlat_grid - 52) <= 1) * (abs(config.mlt_grid - 4) <= .5)
    boundary_good_trough = (abs(config.mlat_grid - 65) < 3) * (abs(config.mlt_grid) >= 10.2)
    boundary_bad_trough = (abs(config.mlat_grid - 52) < 3) * (abs(config.mlt_grid) >= 11.5)
    weird_good_trough = (abs(config.mlat_grid - 40) < 5) * (abs(config.mlt_grid - 9) <= 2.5)
    weird_good_trough += (abs(config.mlat_grid - 34) <= 2) * (abs(config.mlt_grid) > 11.3)
    weird_good_trough += (abs(config.mlat_grid - 44) <= 2) * (abs(config.mlt_grid) > 11.3)
    high_trough = (abs(config.mlat_grid - 80) < 2) * (abs(config.mlt_grid + 6) <= 3)
    arb = np.ones((1, 180)) * 70
    initial_trough = good_trough + small_reject + boundary_good_trough + boundary_bad_trough + weird_good_trough + high_trough
    trough = rbf_inversion.postprocess(initial_trough[None], perimeter_th=50, arb=arb)[0]
    assert trough[good_trough].all()
    assert not trough[small_reject].any()
    assert trough[boundary_good_trough].all()
    assert not trough[boundary_bad_trough].any()
    assert trough[weird_good_trough].all()
    assert not trough[high_trough].any()


def test_get_optimization_args():
    """Verify that get_optimization_args properly handles various outputs
    """
    T = 3
    D = 10
    x = np.random.randn(T, D, D)
    x[0, :2, :2] = np.nan
    times = np.datetime64("2000") + np.arange(T) * np.timedelta64(1, 'h')
    mlt_vals = np.arange(D)
    mlat_grid = np.arange(D)[:, None] * np.ones((1, D))
    arb = np.ones((T, D)) * 7.5
    args = rbf_inversion.get_optimization_args(x, times, mlt_vals, mlat_grid, 10, .5, .5, .5, .5, .5, 1,
                                               'empirical_model', arb, 0)
    var, basis, x_out, tv, l2, t, shp = args[0]
    assert len(args) == T
    assert basis.shape == (D ** 2 - 4, D ** 2)
    assert np.all(x_out == x[0][np.isfinite(x[0])])
    assert np.all(np.diag(tv.toarray()) == 1)
    assert tv.shape == (D ** 2, D ** 2)
    assert l2.min() == .5
    assert l2.max() == 5
    assert shp == (D, D)
    args2 = rbf_inversion.get_optimization_args(x, times, mlt_vals, mlat_grid, 10, .5, .5, .5, .5, .5, 2,
                                                'empirical_model', arb, 0)
    assert np.mean(args2[0][4] != l2) > .75
    args3 = rbf_inversion.get_optimization_args(x, times, mlt_vals, mlat_grid, 10, .5, .5, .5, .5, .5, 1,
                                                'auroral_boundary', arb, 0)
    l = abs(mlat_grid - arb[0]).ravel()
    l -= l.min()
    l = (10 - 1) * l / l.max() + 1
    l *= .5
    assert np.all(l == args3[0][4])


def test_artifacts():
    """This verifies that inputting a 0-360 mlon and 30-60 mlat grid gives you back the artifact grid
    """
    mlt_grid = np.arange(-12, 12, 24/360)[None, :] * np.ones((60, 1))
    mlat_grid = np.arange(30, 90)[:, None] * np.ones((1, 360))
    corr = rbf_inversion.get_artifacts(np.ones(1) * 180, '9', mlt_grid=mlt_grid, mlat_grid=mlat_grid)[0]
    artifacts = np.load(config.artifact_file)
    assert np.allclose(np.roll(artifacts['9'], 180, axis=1), corr)

    corr = rbf_inversion.get_artifacts(np.ones(1) * 179, '7')[0]
    assert np.allclose(np.roll(artifacts['7'], 180, axis=1)[:, ::2], corr)


def test_get_tec_troughs():
    """Verify that get_tec_troughs can detect an actual trough, verify that high troughs are rejected using auroral
    boundary data
    """
    T = 12
    one_h = np.timedelta64(1, 'h')
    start_time = np.datetime64("2015-10-07T06:00:00")
    end_time = start_time + np.timedelta64(T, 'h')
    comparison_times = np.arange(start_time, end_time, one_h)
    bg_est_shape = (3, 19, 19)
    tec_start = comparison_times[0] - np.floor(bg_est_shape[0] / 2) * one_h
    tec_end = comparison_times[-1] + (np.floor(bg_est_shape[0] / 2) + 1) * one_h

    tec, times, ssmlon, n = io.get_tec_data(tec_start, tec_end)
    arb, _ = io.get_arb_data(start_time, end_time)
    trough, x = rbf_inversion.get_tec_troughs(tec, times, bg_est_shape, model_weight_max=5, l2_weight=.1, tv_weight=.05,
                                              tv_hw=2, prior_order=1)
    assert trough.shape == x.shape == (T, 60, 180)
    assert trough[1, 20:30, 60:120].mean() > .5
    assert trough[1][45:].sum() > 100
    trough, x = rbf_inversion.get_tec_troughs(tec, times, bg_est_shape, model_weight_max=5, l2_weight=.1, tv_weight=.05,
                                              tv_hw=2, prior_order=1, arb=arb)
    assert trough[1, 20:30, 60:120].mean() > .5
    assert trough[1][45:].sum() == 0
