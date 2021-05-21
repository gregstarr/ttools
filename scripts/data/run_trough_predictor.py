import numpy as np

from ttools import io, utils, config, trough_labeling


def run_date(date, bg_est_shape=(3, 15, 15), model_weight_max=20, rbf_bw=1, tv_hw=1, tv_vw=1, l2_weight=.1,
             tv_weight=.05, perimeter_th=50, area_th=20, artifact_key=None, auroral_boundary=True, prior_order=1,
             prior='empirical_model', prior_arb_offset=-1):
    # setup time arrays
    time_step = np.timedelta64(1, 'h')
    start_time = date.astype('datetime64[D]').astype('datetime64[s]')
    end_time = start_time + np.timedelta64(1, 'D')
    comparison_times = np.arange(start_time, end_time, time_step)

    # get tec data, run trough detection algo
    tec_start = comparison_times[0] - np.floor(bg_est_shape[0] / 2) * time_step
    tec_end = comparison_times[-1] + (np.floor(bg_est_shape[0] / 2) + 1) * time_step
    tec, times, ssmlon, n = io.get_tec_data(tec_start, tec_end, time_step)
    tec, ssmlon = utils.moving_func_trim(bg_est_shape[0], tec, ssmlon)
    artifacts = None
    arb = None
    if artifact_key is not None:
        artifacts = rbf_inversion.get_artifacts(ssmlon, artifact_key)
    if auroral_boundary:
        arb, _ = io.get_arb_data(start_time, end_time, time_step)
    tec_troughs, x = rbf_inversion.get_tec_troughs(tec, times, bg_est_shape, model_weight_max, rbf_bw, tv_hw, tv_vw,
                                                   l2_weight, tv_weight, perimeter_th, area_th, artifacts, arb,
                                                   prior_order, prior, prior_arb_offset)
    return x, tec_troughs, comparison_times, ssmlon


if __name__ == "__main__":
    output_fn = "E:\\dataset.npz"

    params = {
        'bg_est_shape': (1, 19, 11),
        'model_weight_max': 15,
        'rbf_bw': 1,
        'tv_hw': 2,
        'tv_vw': 1,
        'l2_weight': .05,
        'tv_weight': .15,
        'perimeter_th': 40,
        'area_th': 40,
        'artifact_key': None,
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
    xs = []

    for date in dates:
        x, trough, time, ssmlon = run_date(date, **params)
        troughs.append(trough)
        ssmlons.append(ssmlon)
        times.append(time)
        xs.append(x)

        np.savez(output_fn,
                 x=np.concatenate(xs, axis=0),
                 trough=np.concatenate(troughs, axis=0),
                 ssmlon=np.concatenate(ssmlons, axis=0),
                 time=np.concatenate(times, axis=0))
