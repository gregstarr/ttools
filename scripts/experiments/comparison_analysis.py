import pandas
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from ttools import compare

plt.style.use('seaborn')

results = pandas.read_csv("E:\\temp_comparison_results\\results2.csv")
mlon_mask = ~((results['mlon'] >= 130) & (results['mlon'] <= 260))
mlon_ind, = np.nonzero(mlon_mask.values)
diffs = compare.get_diffs(results[mlon_mask])
statistics = compare.process_results(results, bad_mlon_range=[130, 260])

for k, v in statistics.items():
    print(k, v)

fail_mask = np.any((diffs < diffs.quantile(.01)) + (diffs > diffs.quantile(.99)), axis=1)
fail_ind = fail_mask.index.values[fail_mask.values]

results.iloc[fail_ind]

g = sns.PairGrid(diffs)
g.map_upper(sns.scatterplot, s=2)
g.map_lower(sns.scatterplot, s=2)
g.map_diag(sns.histplot, kde=True)
plt.show()
