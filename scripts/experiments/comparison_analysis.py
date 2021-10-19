import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ttools import compare, convert

plt.style.use('ggplot')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
blue = colors[1]

stats = {}
for threshold in np.linspace(-1, 2, 31):
    results = pandas.read_csv(f"E:\\method_tlm1\\results_{threshold:.2f}.csv")
    mlon_mask = ~((results['mlon'] >= 130) & (results['mlon'] <= 260))
    mlon_ind, = np.nonzero(mlon_mask.values)
    diffs = compare.get_diffs(results[mlon_mask])
    statistics = compare.process_results(results, bad_mlon_range=[130, 260])

    for k, v in statistics.items():
        if k not in stats:
            stats[k] = []
        stats[k].append(v)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(np.linspace(-1, 2, 31), stats['mlon_acc'], label='mlon_acc')
ax[0].plot(np.linspace(-1, 2, 31), stats['mlon_tpr'], label='mlon_tpr')
ax[0].plot(np.linspace(-1, 2, 31), stats['mlon_tnr'], label='mlon_tnr')
ax[0].plot(np.linspace(-1, 2, 31), stats['mlon_fpr'], label='mlon_fpr')
ax[0].plot(np.linspace(-1, 2, 31), stats['mlon_fnr'], label='mlon_fnr')
ax[0].legend()

ax[1].plot(np.linspace(-1, 2, 31), stats['mlon_pwall_diff_mean'], label='mlon_pwall_diff_mean')
ax[1].plot(np.linspace(-1, 2, 31), stats['mlon_pwall_diff_std'], label='mlon_pwall_diff_std')
ax[1].plot(np.linspace(-1, 2, 31), stats['mlon_ewall_diff_mean'], label='mlon_ewall_diff_mean')
ax[1].plot(np.linspace(-1, 2, 31), stats['mlon_ewall_diff_std'], label='mlon_ewall_diff_std')
ax[1].legend()

threshold = 0.0
# results = pandas.read_csv(f"E:\\method_tlm2\\results_{threshold:.2f}.csv")
# results = pandas.read_csv(f"E:\\method_baseline\\results.csv")
# results = pandas.read_csv("E:\\method_constant_baseline\\results.csv")
for fn in ["E:\\method_tlm1_final\\results.csv", "E:\\method_tlm2_final\\results.csv", "E:\\method_baseline_final\\results.csv",
           "E:\\method_constant_baseline\\results.csv", "E:\\method_model_baseline\\results.csv"]:
    print(fn)
    results = pandas.read_csv(fn)
    mlon_mask = ~((results['mlon'] >= 130) & (results['mlon'] <= 260))
    mlon_ind, = np.nonzero(mlon_mask.values)
    diffs = compare.get_diffs(results[mlon_mask])
    statistics = compare.process_results(results, bad_mlon_range=[130, 260])
    for k, v in statistics.items():
        print(k, v)

    g = sns.PairGrid(diffs)
    g.map_upper(sns.scatterplot, s=1, color=blue)
    g.map_diag(sns.histplot, kde=True)

plt.show()
