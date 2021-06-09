import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ttools import compare, convert

plt.style.use('ggplot')

results = pandas.read_csv("E:\\temp_comparison_results\\results.csv")
mlon_mask = ~((results['mlon'] >= 130) & (results['mlon'] <= 260))
mlon_ind, = np.nonzero(mlon_mask.values)
diffs = compare.get_diffs(results[mlon_mask])
statistics = compare.process_results(results, bad_mlon_range=[130, 260])

for k, v in statistics.items():
    print(k, v)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
mlt = convert.mlon_to_mlt_array(results['mlon'].values, results['time'].values)
bins = np.arange(0, 25, 2)
acc = np.zeros(len(bins) - 1)
dw = np.zeros(len(bins) - 1)
dws = np.zeros(len(bins) - 1)
n = np.zeros(len(bins) - 1)
for i in range(len(bins) - 1):
    mask = (mlt >= bins[i]) & (mlt < bins[i + 1])
    acc[i] = np.mean((results['swarm_trough'] == results['tec_trough'])[mask])
    mask1 = (results['tec_trough'] & results['swarm_trough']) & mlon_mask
    mask2 = (mlt[mask1] >= bins[i]) & (mlt[mask1] < bins[i + 1])
    n[i] = mask2.sum()
    dw[i] = np.mean(diffs['pwall_diff'][mask2] - diffs['ewall_diff'][mask2])
    dws[i] = np.std(diffs['pwall_diff'][mask2] - diffs['ewall_diff'][mask2])

mask = n >= 200
x = np.column_stack((bins[:-1], bins[1:])).mean(axis=1)
x[x > 12] -= 24
ax[0].bar(x[mask], acc[mask], 1.8)
ax[1].errorbar(x[mask], dw[mask], yerr=dws[mask], fmt='o')
ax[0].set_xlabel('MLT')
ax[0].set_title('Accuracy')
ax[1].set_xlabel('MLT')
ax[1].set_title('Width Difference')

g = sns.PairGrid(diffs)
g.map_lower(sns.scatterplot, s=2)
g.map_diag(sns.histplot, kde=True)
plt.show()
