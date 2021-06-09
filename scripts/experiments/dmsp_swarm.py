import numpy as np
import pandas
import matplotlib.pyplot as plt

from ttools import satellite, io, utils

dates = utils.get_random_dates(10)
troughs = []
times = []
for date in dates:
    t = np.arange(date, date + np.timedelta64(1, 'D'), np.timedelta64(1, 'h'))
    segments = satellite.get_segments_data(t, 'dmsp')
    trough = satellite.get_troughs(segments)
    if trough['trough'].any():
        t = trough.iloc[np.random.choice(np.arange(trough.shape[0])[trough['trough'].values])]
        d = segments[t['sat']][t['direction']][t['tec_ind']]
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(d['mlat'], d['dne'], '.')
        ax[0].plot(d['mlat'], d['smooth_dne'])
        ax[0].set_title(f"Satellite: {t['sat']} | MLT1: {t['seg_e1_mlt']:.2f} | MLT2: {t['seg_e2_mlt']:.2f}")
        ax[1].plot(d['mlat'], d['hor_ion_v'])
        plt.show()
    troughs.append(trough)
    times.append(t)
troughs = pandas.concat(troughs, ignore_index=True)
troughs = satellite.fix_trough_list(troughs)
