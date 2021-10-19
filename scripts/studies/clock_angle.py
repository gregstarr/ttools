import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import binary_dilation, ball

from ttools import config, io, plotting, utils

import probability as prob_plot


# SIN_MLT_COEF = -1.62729446
# COS_MLT_COEF = -4.11565399
# SIN_SEASON_COEF = 0.67096358
# COS_SEASON_COEF = -2.17226043
# INTERCEPT = 9.12546372639065

SIN_MLT_COEF = 0
COS_MLT_COEF = 0
SIN_SEASON_COEF = 0.9783416
COS_SEASON_COEF = -2.0953347
INTERCEPT = 9.083427842851142


def get_linear_model(times):
    doy_theta = 2 * np.pi * (times - times.astype('datetime64[Y]')).astype('timedelta64[D]').astype(float) / 365
    mlt_theta = np.pi * config.mlt_grid / 12
    tec_mlt = SIN_MLT_COEF * np.sin(mlt_theta) + COS_MLT_COEF * np.cos(mlt_theta)
    tec_season = SIN_SEASON_COEF * np.sin(doy_theta) + COS_SEASON_COEF * np.cos(doy_theta)
    return tec_season[:, None, None] + tec_mlt[None, :, :] + INTERCEPT


def plot_clock_angle_prob_diff(trough, x, imf_masks, delay=1):
    probs = {}
    vm = [0, 0]
    for imf_name, mask in imf_masks.items():
        idx = np.argwhere(mask)[:, 0] + delay
        idx = idx[idx < trough.shape[0]]
        fin = np.isfinite(x[idx])
        ftrough = trough[idx].astype(float)
        ftrough[~fin] = np.nan
        probs[imf_name] = bn.nanmean(ftrough, axis=0)
        probs[imf_name][np.sum(fin, axis=0) < 500] = np.nan
        vm[0] = min(vm[0], np.nanmin(abs(probs[imf_name])))
        vm[1] = max(vm[1], np.nanmax(abs(probs[imf_name])))

    fig = plt.figure(figsize=(18, 6), tight_layout=True)
    gs = plt.GridSpec(1, 4, width_ratios=[30, 30, 30, 1])
    for i, (imf_name, prob) in enumerate(probs.items()):
        ax = fig.add_subplot(gs[i], polar=True)
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, prob, cmap='jet', vmin=vm[0], vmax=vm[1])
        plotting.format_polar_mag_ax(ax, tick_color='grey')
        ax.set_title(imf_name, loc='left')
    cb_ax = fig.add_subplot(gs[-1])
    plt.colorbar(pcm, cax=cb_ax)
    fig.savefig(f"C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\trough_prob_{delay}.png")

    diffs = {}
    vm = 0
    for imf1, imf2 in [['west', 'south'], ['east', 'west'], ['east', 'south']]:
        diff_name = f'{imf1} - {imf2}'
        diffs[diff_name] = probs[imf1] - probs[imf2]
        vm = max(vm, np.nanmax(abs(diffs[diff_name])))

    fig = plt.figure(figsize=(18, 6), tight_layout=True)
    gs = plt.GridSpec(1, 4, width_ratios=[30, 30, 30, 1])
    for i, (diff_name, diff) in enumerate(diffs.items()):
        ax = fig.add_subplot(gs[i], polar=True)
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, diff, cmap='coolwarm', vmin=-1 * vm, vmax=vm)
        plotting.format_polar_mag_ax(ax)
        ax.set_title(diff_name, loc='left')
    cb_ax = fig.add_subplot(gs[-1])
    plt.colorbar(pcm, cax=cb_ax)
    fig.savefig(f"C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\trough_prob_difference_{delay}.png")


def plot_average_profiles(trough, x, imf_masks, delay=1, only_trough=False):
    profs = {}
    vm = 0
    for imf_name, mask in imf_masks.items():
        idx = np.argwhere(mask)[:, 0] + delay
        idx = idx[idx < trough.shape[0]]
        if only_trough:
            trough_x = np.where(np.broadcast_to(np.any(trough[idx], axis=1)[:, None, :], x[idx].shape), x[idx], np.nan)
            profs[imf_name] = np.nanmean(trough_x, axis=0)
            profs[imf_name][np.sum(np.isfinite(trough_x), axis=0) < 100] = np.nan
        else:
            profs[imf_name] = np.nanmean(x[idx], axis=0)
            profs[imf_name][np.sum(np.isfinite(x[idx]), axis=0) < 100] = np.nan
        vm = max(vm, np.nanmax(abs(profs[imf_name])))

    fig = plt.figure(figsize=(18, 6), tight_layout=True)
    gs = plt.GridSpec(1, 4, width_ratios=[30, 30, 30, 1])
    for i, (imf_name, prof) in enumerate(profs.items()):
        ax = fig.add_subplot(gs[i], polar=True)
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, prof, cmap='coolwarm', vmin=-vm, vmax=vm)
        plotting.format_polar_mag_ax(ax)
        ax.set_title(imf_name, loc='left')
    cb_ax = fig.add_subplot(gs[-1])
    plt.colorbar(pcm, cax=cb_ax)
    if only_trough:
        fig.savefig(f"C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\average_x_trough_profile_{delay}.png")
    else:
        fig.savefig(f"C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\average_x_profile_{delay}.png")

    diffs = {}
    vm = 0
    for imf1, imf2 in [['west', 'south'], ['east', 'west'], ['east', 'south']]:
        diff_name = f'{imf1} - {imf2}'
        diffs[diff_name] = profs[imf1] - profs[imf2]
        vm = max(vm, np.nanmax(abs(diffs[diff_name])))

    fig = plt.figure(figsize=(18, 6), tight_layout=True)
    gs = plt.GridSpec(1, 4, width_ratios=[30, 30, 30, 1])
    for i, (diff_name, diff) in enumerate(diffs.items()):
        ax = fig.add_subplot(gs[i], polar=True)
        pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, diff, cmap='coolwarm', vmin=-1 * vm, vmax=vm)
        plotting.format_polar_mag_ax(ax)
        ax.set_title(diff_name, loc='left')
    cb_ax = fig.add_subplot(gs[-1])
    plt.colorbar(pcm, cax=cb_ax)
    if only_trough:
        fig.savefig(f"C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\average_x_trough_profile_difference_{delay}.png")
    else:
        fig.savefig(f"C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\average_x_profile_difference_{delay}.png")


if __name__ == "__main__":
    # Load trough dataset
    trough_data = np.load("E:\\dataset.npz")
    trough = trough_data['trough']
    x = trough_data['x']
    xi = x.copy()
    xi[~trough] = np.inf
    min_mlat = config.mlat_vals[np.argmin(xi, axis=1)]
    depth = np.min(xi, axis=1)
    min_mlat[~np.isfinite(depth)] = np.nan
    depth[~np.isfinite(depth)] = np.nan
    omni = io.get_omni_data()
    bmag = omni['b_mag'][trough_data['time']].values
    # e_field = omni['e_field'][trough_data['time']].values
    by = omni['by_gsm'][trough_data['time']].values
    bz = omni['bz_gsm'][trough_data['time']].values
    bz += np.random.randn(*bz.shape) * .1
    clock_angle = np.arctan2(by, bz)
    clock_angle[clock_angle < 0] += 2 * np.pi
    clock_angle = 180 * clock_angle / np.pi
    kp = io.get_kp(trough_data['time'])
    bmag_mask = (bmag >= 2) & (bmag <= 6)
    angle_radius = 30
    imf_masks = {
        'west': (abs(clock_angle - 270) < angle_radius) & bmag_mask,
        'south': (abs(clock_angle - 180) < angle_radius) & bmag_mask,
        'east': (abs(clock_angle - 90) < angle_radius) & bmag_mask,
    }
    bmidx = np.argwhere(bmag_mask)[:, 0]
    bmidx = bmidx[bmidx < trough.shape[0] - 1]
    prob_plot.plot_param_mlat(trough[bmidx + 1], bz[bmidx], np.isfinite(x[bmidx + 1]), name='B_z', mlt_center=3, save_dir="C:\\Users\\Greg\\Desktop\\study plots\\solar wind study")
    prob_plot.plot_param_min_mlat(clock_angle[bmidx], min_mlat[bmidx + 1], 0, 2, name='clock angle', lr=False, save_path="C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\clock_angle_mlat.png")
    prob_plot.plot_param_min_mlat(bz[bmidx], min_mlat[bmidx + 1], 0, 2, name='B_z', lr=True, save_path="C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\bz_mlat.png")
    plot_average_profiles(trough, x, imf_masks, delay=1)
    plot_average_profiles(trough, x, imf_masks, delay=1, only_trough=True)
    plot_clock_angle_prob_diff(trough, x, imf_masks, delay=1)

    tec_prof_data = {
        exp: {val: np.zeros((3, 60)) for val in ['sum', 'n']} for exp in
        ['evening_all', 'evening_trough', 'evening_not_trough', 'morning_all', 'morning_trough', 'morning_not_trough',]
    }
    tec_avg_data = {
        exp: {val: np.zeros((3, 60, 180)) for val in ['sum', 'n']} for exp in
        ['all', 'trough_profile', 'not_trough_profile', 'trough_local', 'not_trough_local', 'trough_global', 'not_trough_global', 'high_kp', 'low_kp']
    }
    flow_avg_data = {
        exp: {coord: {val: np.zeros((3, 60, 180)) for val in ['sum', 'n']} for coord in ['x', 'y']} for exp in
        ['all', 'trough_global', 'not_trough_global', 'trough_local', 'not_trough_local', 'high_kp', 'low_kp']
    }
    evening_mlt_mask = abs(config.mlt_vals + 6) < 2
    morning_mlt_mask = abs(config.mlt_vals - 6) < 2
    batch_size = 4000
    for batch in range(int(np.ceil(trough.shape[0] / batch_size))):
        print(batch, int(np.ceil(trough.shape[0] / batch_size)))
        sl = slice(batch * batch_size, min((batch + 1) * batch_size, trough.shape[0]))
        l = sl.stop - sl.start
        start_time = trough_data['time'][batch * batch_size]
        end_time = min(start_time + (batch_size - 1) * np.timedelta64(1, 'h'), trough_data['time'][-1]) + np.timedelta64(1, 'h')
        tec, times, *_ = io.get_tec_data(start_time, end_time)
        tec -= get_linear_model(times)
        tr = trough[sl]
        trough_tec = np.where(np.broadcast_to(np.any(tr, axis=1)[:, None, :], tec.shape), tec, np.nan)
        not_trough_tec = np.where(np.broadcast_to(~np.any(tr, axis=1)[:, None, :], tec.shape), tec, np.nan)
        fx, fy, _ = io.get_superdarn_data(start_time, end_time)

        sidelen = 15
        ex_tr = binary_dilation(np.pad(tr, ((0, 0), (0, 0), (sidelen // 2, sidelen // 2)), 'wrap'), np.ones((1, sidelen, sidelen)))
        ex_tr = ex_tr[:, :, sidelen // 2:-(sidelen // 2)]

        for i, masks in enumerate(imf_masks.values()):
            mask = masks[sl]
            idx = np.argwhere(mask)[:, 0]
            idx = idx[idx < l - 1]

            tec_prof_data['evening_all']['sum'][i] += np.nansum(tec[:, :, evening_mlt_mask][idx + 1], axis=(0, 2))
            tec_prof_data['evening_all']['n'][i] += np.nansum(np.isfinite(tec[:, :, evening_mlt_mask][idx + 1]), axis=(0, 2))
            tec_prof_data['evening_trough']['sum'][i] += np.nansum(trough_tec[:, :, evening_mlt_mask][idx + 1], axis=(0, 2))
            tec_prof_data['evening_trough']['n'][i] += np.nansum(np.isfinite(trough_tec[:, :, evening_mlt_mask][idx + 1]), axis=(0, 2))
            tec_prof_data['evening_not_trough']['sum'][i] += np.nansum(not_trough_tec[:, :, evening_mlt_mask][idx + 1], axis=(0, 2))
            tec_prof_data['evening_not_trough']['n'][i] += np.nansum(np.isfinite(not_trough_tec[:, :, evening_mlt_mask][idx + 1]), axis=(0, 2))
            tec_prof_data['morning_all']['sum'][i] += np.nansum(tec[:, :, morning_mlt_mask][idx + 1], axis=(0, 2))
            tec_prof_data['morning_all']['n'][i] += np.nansum(np.isfinite(tec[:, :, morning_mlt_mask][idx + 1]), axis=(0, 2))
            tec_prof_data['morning_trough']['sum'][i] += np.nansum(trough_tec[:, :, morning_mlt_mask][idx + 1], axis=(0, 2))
            tec_prof_data['morning_trough']['n'][i] += np.nansum(np.isfinite(trough_tec[:, :, morning_mlt_mask][idx + 1]), axis=(0, 2))
            tec_prof_data['morning_not_trough']['sum'][i] += np.nansum(not_trough_tec[:, :, morning_mlt_mask][idx + 1], axis=(0, 2))
            tec_prof_data['morning_not_trough']['n'][i] += np.nansum(np.isfinite(not_trough_tec[:, :, morning_mlt_mask][idx + 1]), axis=(0, 2))

            tec_avg_data['all']['sum'][i] += np.nansum(tec[idx + 1], axis=0)
            tec_avg_data['all']['n'][i] += np.nansum(np.isfinite(tec[idx + 1]), axis=0)
            tec_avg_data['trough_profile']['sum'][i] += np.nansum(trough_tec[idx + 1], axis=0)
            tec_avg_data['trough_profile']['n'][i] += np.nansum(np.isfinite(trough_tec[idx + 1]), axis=0)
            tec_avg_data['not_trough_profile']['sum'][i] += np.nansum(not_trough_tec[idx + 1], axis=0)
            tec_avg_data['not_trough_profile']['n'][i] += np.nansum(np.isfinite(not_trough_tec[idx + 1]), axis=0)
            tec_avg_data['trough_local']['sum'][i] += np.nansum(np.where(ex_tr[idx + 1], tec[idx], np.nan), axis=0)
            tec_avg_data['trough_local']['n'][i] += np.nansum(np.isfinite(np.where(ex_tr[idx + 1], tec[idx], np.nan)), axis=0)
            tec_avg_data['not_trough_local']['sum'][i] += np.nansum(np.where(~ex_tr[idx + 1], tec[idx], np.nan), axis=0)
            tec_avg_data['not_trough_local']['n'][i] += np.nansum(np.isfinite(np.where(~ex_tr[idx + 1], tec[idx], np.nan)), axis=0)
            m = tr[idx + 1].sum(axis=(1, 2)) > 500
            tec_avg_data['trough_global']['sum'][i] += np.nansum(tec[idx][m], axis=0)
            tec_avg_data['trough_global']['n'][i] += np.nansum(np.isfinite(tec[idx][m]), axis=0)
            m = tr[idx + 1].sum(axis=(1, 2)) < 100
            tec_avg_data['not_trough_global']['sum'][i] += np.nansum(tec[idx][m], axis=0)
            tec_avg_data['not_trough_global']['n'][i] += np.nansum(np.isfinite(tec[idx][m]), axis=0)
            high_kp_mask = kp[sl][idx + 1] > 3
            tec_avg_data['high_kp']['sum'][i] += np.nansum(tec[idx + 1][high_kp_mask], axis=0)
            tec_avg_data['high_kp']['n'][i] += np.nansum(np.isfinite(tec[idx + 1][high_kp_mask]), axis=0)
            low_kp_mask = kp[sl][idx + 1] < 2
            tec_avg_data['low_kp']['sum'][i] += np.nansum(tec[idx + 1][low_kp_mask], axis=0)
            tec_avg_data['low_kp']['n'][i] += np.nansum(np.isfinite(tec[idx + 1][low_kp_mask]), axis=0)

            flow_avg_data['all']['x']['sum'][i] += np.nansum(fx[idx], axis=0)
            flow_avg_data['all']['x']['n'][i] += np.nansum(np.isfinite(fx[idx]), axis=0)
            flow_avg_data['all']['y']['sum'][i] += np.nansum(fy[idx], axis=0)
            flow_avg_data['all']['y']['n'][i] += np.nansum(np.isfinite(fy[idx]), axis=0)
            m = tr[idx + 1].sum(axis=(1, 2)) > 500
            flow_avg_data['trough_global']['x']['sum'][i] += np.nansum(fx[idx][m], axis=0)
            flow_avg_data['trough_global']['x']['n'][i] += np.nansum(np.isfinite(fx[idx][m]), axis=0)
            flow_avg_data['trough_global']['y']['sum'][i] += np.nansum(fy[idx][m], axis=0)
            flow_avg_data['trough_global']['y']['n'][i] += np.nansum(np.isfinite(fy[idx][m]), axis=0)
            m = tr[idx + 1].sum(axis=(1, 2)) < 100
            flow_avg_data['not_trough_global']['x']['sum'][i] += np.nansum(fx[idx][m], axis=0)
            flow_avg_data['not_trough_global']['x']['n'][i] += np.nansum(np.isfinite(fx[idx][m]), axis=0)
            flow_avg_data['not_trough_global']['y']['sum'][i] += np.nansum(fy[idx][m], axis=0)
            flow_avg_data['not_trough_global']['y']['n'][i] += np.nansum(np.isfinite(fy[idx][m]), axis=0)
            flow_avg_data['trough_local']['x']['sum'][i] += np.nansum(np.where(ex_tr[idx + 1], fx[idx], np.nan), axis=0)
            flow_avg_data['trough_local']['x']['n'][i] += np.nansum(np.isfinite(np.where(ex_tr[idx + 1], fx[idx], np.nan)), axis=0)
            flow_avg_data['trough_local']['y']['sum'][i] += np.nansum(np.where(ex_tr[idx + 1], fy[idx], np.nan), axis=0)
            flow_avg_data['trough_local']['y']['n'][i] += np.nansum(np.isfinite(np.where(ex_tr[idx + 1], fy[idx], np.nan)), axis=0)
            flow_avg_data['not_trough_local']['x']['sum'][i] += np.nansum(np.where(~ex_tr[idx + 1], fx[idx], np.nan), axis=0)
            flow_avg_data['not_trough_local']['x']['n'][i] += np.nansum(np.isfinite(np.where(~ex_tr[idx + 1], fx[idx], np.nan)), axis=0)
            flow_avg_data['not_trough_local']['y']['sum'][i] += np.nansum(np.where(~ex_tr[idx + 1], fy[idx], np.nan), axis=0)
            flow_avg_data['not_trough_local']['y']['n'][i] += np.nansum(np.isfinite(np.where(~ex_tr[idx + 1], fy[idx], np.nan)), axis=0)
            flow_avg_data['high_kp']['x']['sum'][i] += np.nansum(fx[idx][high_kp_mask], axis=0)
            flow_avg_data['high_kp']['x']['n'][i] += np.nansum(np.isfinite(fx[idx][high_kp_mask]), axis=0)
            flow_avg_data['high_kp']['y']['sum'][i] += np.nansum(fy[idx][high_kp_mask], axis=0)
            flow_avg_data['high_kp']['y']['n'][i] += np.nansum(np.isfinite(fy[idx][high_kp_mask]), axis=0)
            flow_avg_data['low_kp']['x']['sum'][i] += np.nansum(fx[idx][low_kp_mask], axis=0)
            flow_avg_data['low_kp']['x']['n'][i] += np.nansum(np.isfinite(fx[idx][low_kp_mask]), axis=0)
            flow_avg_data['low_kp']['y']['sum'][i] += np.nansum(fy[idx][low_kp_mask], axis=0)
            flow_avg_data['low_kp']['y']['n'][i] += np.nansum(np.isfinite(fy[idx][low_kp_mask]), axis=0)

    for exp_name, data in tec_prof_data.items():
        avg_prof = data['sum'] / data['n']
        avg_prof[data['n'] < 100] = np.nan
        fig, ax = plt.subplots(1, 2, sharex=True, tight_layout=True, figsize=(8, 4))
        for i, imf_name in enumerate(imf_masks):
            ax[0].plot(config.mlat_vals, avg_prof[i], label=imf_name)
        ax[0].legend()
        ax[0].set_title(f'{exp_name} Average TEC Profiles')
        ax[0].set_ylabel('TECu')
        ax[0].set_xlabel('MLAT')
        ax[0].grid()

        diffs = {}
        vm = 0
        for imf1, imf2 in [[0, 1], [2, 0], [2, 1]]:
            imf_name_1 = list(imf_masks.keys())[imf1]
            imf_name_2 = list(imf_masks.keys())[imf2]
            diff_name = f'{imf_name_1} - {imf_name_2}'
            ax[1].plot(config.mlat_vals, avg_prof[imf1] - avg_prof[imf2], label=diff_name)
        ax[1].legend()
        ax[1].set_title('TEC Profile Differences')
        ax[1].set_xlabel('MLAT')
        ax[1].set_ylabel('TECu')
        ax[1].grid()

        fig.savefig(f"C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\average_tec_profiles_{exp_name}.png")

    flow_plots = [
        ('all', 'all'),
        ('trough_local', 'trough_local'),
        ('not_trough_local', 'not_trough_local'),
        ('trough_global', 'trough_global'),
        ('not_trough_global', 'not_trough_global'),
        ('trough_local', 'trough_profile'),
        ('trough_global', 'trough_profile'),
        ('high_kp', 'high_kp'),
        ('low_kp', 'low_kp'),
    ]
    tec = {}
    vm = [0, 0]
    for name, tec_data in tec_avg_data.items():
        tec[name] = tec_data['sum'] / tec_data['n']
        tec[name][tec_data['n'] < 100] = np.nan
        vm[0] = min(vm[0], np.nanmin(tec[name]))
        vm[1] = max(vm[1], np.nanmax(tec[name]))

    for flow_exp_name, tec_exp_name in flow_plots:
        flow_data = flow_avg_data[flow_exp_name]
        fx_avg = flow_data['x']['sum'] / flow_data['x']['n']
        fx_avg[flow_data['x']['n'] < 100] = np.nan
        fy_avg = flow_data['y']['sum'] / flow_data['y']['n']
        fy_avg[flow_data['y']['n'] < 100] = np.nan

        fig = plt.figure(tight_layout=True, figsize=(15, 5))
        gs = plt.GridSpec(1, 4, width_ratios=[30, 30, 30, 1])
        for i, imf_name in enumerate(imf_masks):
            ax = fig.add_subplot(gs[i], polar=True)
            pcm = plotting.polar_pcolormesh(ax, config.mlat_grid, config.mlt_grid, tec[tec_exp_name][i], cmap='jet', vmin=vm[0], vmax=vm[1], alpha=.6, edgecolors='none', linewidth=0, antialiased=True)
            q = ax.quiver(np.pi * (config.mlt_grid[::2, ::4] - 6) / 12, 90 - config.mlat_grid[::2, ::4], fx_avg[i, ::2, ::4], fy_avg[i, ::2, ::4], scale=4000, headwidth=1.2)
            plt.quiverkey(q, .9, .9, 300, '300 m/s')
            ax.set_title(imf_name, loc='left')
            plotting.format_polar_mag_ax(ax)
        plt.colorbar(pcm, cax=fig.add_subplot(gs[-1]))
        fig.suptitle(f"convection: {flow_exp_name}, tec: {tec_exp_name}")
        fig.savefig(f"C:\\Users\\Greg\\Desktop\\study plots\\solar wind study\\convection_{flow_exp_name}__tec_{tec_exp_name}.png")

    plt.show()
