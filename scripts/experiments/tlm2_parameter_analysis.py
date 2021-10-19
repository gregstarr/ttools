import pandas
import yaml
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

params = []
i = 0
while True:
    try:
        with open(f"E:\\trough_comparison\\experiment_{i}\\params.yaml") as f:
            params.append(yaml.load(f))
        i += 1
    except:
        break

params = pandas.DataFrame(params)

results_fn = "E:\\trough_comparison\\results.csv"
results = pandas.read_csv(results_fn)
print()
plt.show()
