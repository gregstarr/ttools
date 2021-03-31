import numpy as np

from ttools import io, rbf_inversion, utils


def run_date(date, bg_est_shape=(3, 15, 15), model_weight_max=20, rbf_bw=1, tv_hw=1, tv_vw=1, l2_weight=.1,
             tv_weight=.05, perimeter_th=50, area_th=20, artifact_key=None, auroral_boundary=True, prior_order=1,
             prior='empirical_model', prior_arb_offset=-1):
    # setup time arrays
    one_h = np.timedelta64(1, 'h')
    start_time = date.astype('datetime64[D]').astype('datetime64[s]')
    end_time = start_time + np.timedelta64(1, 'D')
    comparison_times = np.arange(start_time, end_time, one_h)

    # get tec data, run trough detection algo
    tec_start = comparison_times[0] - np.floor(bg_est_shape[0] / 2) * one_h
    tec_end = comparison_times[-1] + (np.floor(bg_est_shape[0] / 2) + 1) * one_h
    tec, times, ssmlon, n = io.get_tec_data(tec_start, tec_end)
    ssmlon, = utils.moving_func_trim(bg_est_shape[0], ssmlon)
    artifacts = None
    arb = None
    if artifact_key is not None:
        artifacts = rbf_inversion.get_artifacts(ssmlon, artifact_key)
    if auroral_boundary:
        arb, _ = io.get_arb_data(start_time, end_time)
    tec_troughs, x = rbf_inversion.get_tec_troughs(tec, times, bg_est_shape, model_weight_max, rbf_bw, tv_hw, tv_vw,
                                                   l2_weight, tv_weight, perimeter_th, area_th, artifacts, arb,
                                                   prior_order, prior, prior_arb_offset)
    return tec_troughs, comparison_times, ssmlon


if __name__ == "__main__":
    output_fn = "E:\\labels.npz"

    params = {
        'bg_est_shape': (1, 17, 17),
        'model_weight_max': 15,
        'rbf_bw': 1,
        'tv_hw': 2,
        'tv_vw': 1,
        'l2_weight': .05,
        'tv_weight': .15,
        'perimeter_th': 40,
        'area_th': 40,
        'artifact_key': '7',
        'auroral_boundary': True,
        'prior_order': 1,
        'prior': 'auroral_boundary',
        'prior_arb_offset': -3
    }

    start_date = np.datetime64("2010-01-01T00:00:00")
    end_date = np.datetime64("2020-01-01T00:00:00")
    one_day = np.timedelta64(1, 'D')
    dates = np.arange(start_date, end_date, one_day)

    troughs = []
    ssmlons = []
    times = []

    for date in dates:
        trough, time, ssmlon = run_date(date, **params)
        troughs.append(trough)
        ssmlons.append(ssmlon)
        times.append(time)

        np.savez(output_fn,
                 trough=np.concatenate(troughs, axis=0),
                 ssmlon=np.concatenate(ssmlons, axis=0),
                 time=np.concatenate(times, axis=0))
