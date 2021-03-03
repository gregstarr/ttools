"""
Experiment 1: swarm tec correlation
    - for various background estimation sizes and artifact keys:
        - collect random days
        - get dtec prof
        - interpolate swarm dne at the profile points
        - estimate mean and covariance between the two
"""
import numpy as np
import pandas

from ttools import io, rbf_inversion, swarm, utils, config


def run_experiment(n, bg_est_shape, artifact_key):
    start_date = np.datetime64("2014-01-01")
    end_date = np.datetime64("2020-01-01")
    time_range_days = (end_date - start_date).astype('timedelta64[D]').astype(int)
    offsets = np.random.randint(0, time_range_days, n)
    dates = start_date + offsets.astype('timedelta64[D]')

    x = []
    dne = []
    mlat_x = []
    mlat_dne = []
    for date in dates:
        _x, _dne, _mlat_x, _mlat_dne = run_day(date, bg_est_shape, artifact_key)
        x += _x
        dne += _dne
        mlat_x += _mlat_x
        mlat_dne += _mlat_dne

    x = np.concatenate(x, axis=0)
    dne = np.concatenate(dne, axis=0)
    mlat_x = np.array(mlat_x)
    mlat_dne = np.array(mlat_dne)

    data = np.column_stack((x, dne))
    mean = np.nanmean(data, axis=0)
    cov = pandas.DataFrame(data=data).corr().values

    mlat_data = np.column_stack((mlat_x, mlat_dne))
    mlat_mean = np.nanmean(mlat_data, axis=0)
    mlat_cov = pandas.DataFrame(data=mlat_data).cov().values
    return mean, cov, mlat_mean, mlat_cov


def run_day(date, bg_est_shape, artifact_key):
    print(f"Running {date}")
    one_h = np.timedelta64(1, 'h')
    start_time = date.astype('datetime64[D]').astype('datetime64[s]')
    end_time = start_time + np.timedelta64(1, 'D')
    comparison_times = np.arange(start_time, end_time, one_h)

    swarm_segments = swarm.get_segments_data(comparison_times)
    swarm_troughs = swarm.get_swarm_troughs(swarm_segments)

    tec_start = comparison_times[0] - np.floor(bg_est_shape[0] / 2) * one_h
    tec_end = comparison_times[-1] + (np.floor(bg_est_shape[0] / 2) + 1) * one_h
    tec, times, ssmlon, _ = io.get_tec_data(tec_start, tec_end)
    x, times = rbf_inversion.preprocess_interval(tec, times, bg_est_shape=bg_est_shape)
    utils.moving_func_trim(bg_est_shape[0], ssmlon)
    if artifact_key is not None:
        x -= rbf_inversion.get_artifacts(ssmlon, artifact_key)
    data_grids = [config.mlat_grid, x[swarm_troughs['tec_ind']]]
    mlat_profs, x_profs = utils.get_grid_slice_line(swarm_troughs['seg_e1_mlt'], swarm_troughs['seg_e1_mlat'],
                                                    swarm_troughs['seg_e2_mlt'], swarm_troughs['seg_e2_mlat'],
                                                    data_grids, config.mlt_grid, config.mlat_grid, linewidth=3)
    t1 = swarm_troughs['seg_e1_mlt'].values * np.pi / 12
    t2 = swarm_troughs['seg_e2_mlt'].values * np.pi / 12
    seg_mlt = utils.average_angles(t1, t2) * 24 / np.pi
    mlon = (15 * seg_mlt - 180 + ssmlon[swarm_troughs['tec_ind']] + 360) % 360
    mlon_mask = ~((mlon >= 130) & (mlon <= 260))
    x_list = []
    dne_list = []
    swarm_min_mlat = []
    tec_min_mlat = []
    for i, row in swarm_troughs[mlon_mask].iterrows():
        segment = swarm_segments[row['sat']][row['direction']][row['tec_ind']]
        idx = np.argsort(segment['mlat'])
        smooth_dne = np.interp(mlat_profs[i], segment['mlat'][idx], segment['smooth_dne'][idx])
        x_list.append(x_profs[i])
        dne_list.append(smooth_dne)
        if row['trough'] and np.isnan(x_profs[i]).mean() < .75:
            swarm_min_mlat.append(row['min_mlat'])
            tec_min_mlat.append(mlat_profs[i][np.nanargmin(x_profs[i])])
    return x_list, dne_list, tec_min_mlat, swarm_min_mlat


if __name__ == "__main__":
    import itertools
    import matplotlib.pyplot as plt

    bg_sizes = [13, 15, 17, 19, 21]
    artifact_keys = [None, '3', '5', '7', '9']

    data = []
    mlat_data = []

    n = 20
    for i, (bg_size, artifact_key) in enumerate(itertools.product(bg_sizes, artifact_keys)):
        print(bg_size, artifact_key)
        mean, cov, mlat_mean, mlat_cov = run_experiment(n, (1, bg_size, bg_size), artifact_key)
        data.append({'m0': mean[0], 'm1': mean[1], 's00': cov[0, 0], 's01': cov[0, 1], 's11': cov[1, 1],
                     'bg_size': bg_size, 'artifact_key': artifact_key})
        mlat_data.append({'m0': mlat_mean[0], 'm1': mlat_mean[1], 's00': mlat_cov[0, 0], 's01': mlat_cov[0, 1],
                          's11': mlat_cov[1, 1], 'bg_size': bg_size, 'artifact_key': artifact_key})
    data = pandas.DataFrame(data)
    mlat_data = pandas.DataFrame(mlat_data)
    print(data)
    print(mlat_data)

    corr = data['s01'].values.reshape((len(bg_sizes), len(artifact_keys)))
    plt.figure()
    plt.imshow(corr, cmap='Blues')
    plt.colorbar()
    plt.xticks(np.arange(len(artifact_keys)), ['None', '3', '5', '7', '9'])
    plt.xlabel('Artifact removal filter size')
    plt.yticks(np.arange(len(bg_sizes)), bg_sizes)
    plt.ylabel('Background estimation filter size')

    mlat_cov = mlat_data['s01'].values.reshape((len(bg_sizes), len(artifact_keys)))
    plt.figure()
    plt.imshow(mlat_cov, cmap='Blues')
    plt.colorbar()
    plt.xticks(np.arange(len(artifact_keys)), ['None', '3', '5', '7', '9'])
    plt.xlabel('Artifact removal filter size')
    plt.yticks(np.arange(len(bg_sizes)), bg_sizes)
    plt.ylabel('Background estimation filter size')

    mlat_corr = (mlat_data['s01'] / (np.sqrt(mlat_data['s00']) * np.sqrt(mlat_data['s11']))).values.reshape(
        (len(bg_sizes), len(artifact_keys)))
    plt.figure()
    plt.imshow(mlat_corr, cmap='Blues')
    plt.colorbar()
    plt.xticks(np.arange(len(artifact_keys)), ['None', '3', '5', '7', '9'])
    plt.xlabel('Artifact removal filter size')
    plt.yticks(np.arange(len(bg_sizes)), bg_sizes)
    plt.ylabel('Background estimation filter size')

    plt.show()
