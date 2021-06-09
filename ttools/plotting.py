import numpy as np

from ttools import convert


def format_polar_mag_ax(ax, tick_color='black'):
    if isinstance(ax, np.ndarray):
        for a in ax.flatten():
            format_polar_mag_ax(a)
    else:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                       labeltop=False, labelleft=False, labelright=False)
        ax.set_ylim(0, 60)
        ax.set_xticks(np.arange(8) * np.pi/4)
        ax.set_xticklabels((np.arange(8) * 3 + 6) % 24)
        ax.set_yticks([10, 20, 30, 40, 50])
        ax.set_yticklabels([80, 70, 60, 50, 40])
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, colors=tick_color)
        ax.tick_params(axis='y', which='both', left=True, labelleft=True, width=0, length=0, colors=tick_color)
        ax.set_rlabel_position(80)
        ax.grid(True)


def plot_swarm_troughs_line(ax, mlat, dne, smooth_dne, trough):
    ax.plot(mlat, smooth_dne)
    ax.plot(mlat, dne, 'k.', ms=1)
    if trough['trough']:
        ax.plot(trough[['e1_mlat', 'e2_mlat']], [0, 0], 'r', linestyle='solid', marker='|', ms=10)
        ax.plot(trough['min_mlat'], trough['min_dne'], 'rx')


def plot_swarm_troughs_polar(ax, troughs):
    for i, trough in troughs.iterrows():
        ax.plot((trough[['seg_e1_mlt', 'seg_e2_mlt']] - 6) * np.pi / 12,
                90 - trough[['seg_e1_mlat', 'seg_e2_mlat']],
                'k--')
        if trough['trough']:
            ax.plot((trough[['e1_mlt', 'e2_mlt']] - 6) * np.pi / 12, 90 - trough[['e1_mlat', 'e2_mlat']], 'r-')


def polar_pcolormesh(ax, mlat, mlt, value, cs='mlt', **kwargs):
    r = 90 - mlat
    if cs == 'mlt':
        t = (mlt - 6) * np.pi / 12
    elif cs == 'mlon':
        t = mlt * np.pi / 180
    else:
        raise Exception("Not recognized coordinate system, choose 'mlon' or 'mlt'")
    return ax.pcolormesh(t - np.pi / 180, r + .5, value, shading='auto', **kwargs)


def plot_mlon_lines(ax, ssmlon, mlons=np.arange(0, 360, 45)):
    for mlon in mlons:
        mlt = convert.mlon_to_mlt_sub(mlon, ssmlon)
        theta = np.pi * (mlt - 6) / 12
        ax.plot([theta, theta], [0, 60], '--', color='grey')
        ax.annotate(str(mlon), (theta, 58))


def prepare_swarm_line_plot(t, swarm_segments, swarm_troughs, mlat_profs, x_profs):
    troughs = swarm_troughs[swarm_troughs['tec_ind'] == t]
    m_p = [p for i, p in enumerate(mlat_profs) if swarm_troughs['tec_ind'].values[i] == t]
    x_p = [p for i, p in enumerate(x_profs) if swarm_troughs['tec_ind'].values[i] == t]
    segments = []
    for sat_segments in swarm_segments.values():
        for segment in sat_segments.values():
            segments.append(segment[t])
    return troughs, segments, m_p, x_p


def plot_arb(ax, mlt, arb_mlat):
    theta = np.pi * (mlt - 6) / 12
    r = 90 - arb_mlat
    ax.plot(theta, r, 'k--')


def plot_all(polar_ax, line_ax, mlat_grid, mlt_grid, tec, x, swarm_troughs, tec_trough, swarm_segments, mlat_profs,
             x_profs, ssmlon, arb=None):
    polar_pcolormesh(polar_ax[0], mlat_grid, mlt_grid, tec, vmin=0, vmax=20)
    plot_swarm_troughs_polar(polar_ax[0], swarm_troughs)
    plot_mlon_lines(polar_ax[0], ssmlon)
    polar_pcolormesh(polar_ax[1], mlat_grid, mlt_grid, x, vmin=-.5, vmax=.5, cmap='coolwarm')
    plot_swarm_troughs_polar(polar_ax[1], swarm_troughs)
    plot_mlon_lines(polar_ax[1], ssmlon)
    polar_pcolormesh(polar_ax[2], mlat_grid, mlt_grid, tec_trough, vmin=0, vmax=1, cmap='Blues')
    plot_swarm_troughs_polar(polar_ax[2], swarm_troughs)
    plot_mlon_lines(polar_ax[2], ssmlon)
    if arb is not None:
        plot_arb(polar_ax[0], mlt_grid[0], arb)
        plot_arb(polar_ax[1], mlt_grid[0], arb)
    format_polar_mag_ax(polar_ax)

    for i, (segment, ax, mlat_prof, x_prof) in enumerate(zip(swarm_segments, line_ax.flatten(), mlat_profs, x_profs)):
        plot_swarm_troughs_line(ax, segment['mlat'], segment['dne'], segment['smooth_dne'], swarm_troughs.iloc[i])
        ax.plot(mlat_prof, x_prof, 'g--')
        ax.set_title(f"mlt: {swarm_troughs.iloc[i]['seg_e1_mlt']}, sat: {swarm_troughs.iloc[i]['sat']}")
        ax.grid(True)
