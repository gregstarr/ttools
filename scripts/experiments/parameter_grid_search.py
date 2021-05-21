import pandas
import yaml
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

params = []
for i in range(200):
    try:
        with open(f"E:\\trough_comparison\\experiment_{i}\\params.yaml") as f:
            params.append(yaml.load(f))
    except:
        break

params = pandas.DataFrame(params)

results_fn = "E:\\trough_comparison\\results.csv"
results = pandas.read_csv(results_fn)
w = -(results['mlon_pwall_diff_mean'] - results['mlon_ewall_diff_mean'])
o = -.5 * (results['mlon_ewall_diff_mean'] + results['mlon_pwall_diff_mean'])
s = np.hypot(results['mlon_ewall_diff_std'], results['mlon_pwall_diff_std'])

fig, ax = plt.subplots(1, 4, figsize=(16, 6), tight_layout=True)
fig.suptitle("L2 Weight")
for i, v in enumerate(np.unique(params['l2_weight'])):
    mask = params['l2_weight'] == v
    ax[0].plot(results['mlon_acc'][mask], w[mask], '.', color=cmap[i], label=v)
    ax[1].plot(results['mlon_acc'][mask], o[mask], '.', color=cmap[i], label=v)
    ax[2].plot(results['mlon_acc'][mask], s[mask], '.', color=cmap[i], label=v)
    ax[3].plot(results['mlon_fpr'][mask], results['mlon_tpr'][mask], '.', color=cmap[i], label=v)
ax[0].set_xlabel('Accuracy')
ax[0].set_title('Lat width difference (TEC - SWARM)')
ax[1].set_xlabel('Accuracy')
ax[1].set_title('Lat Offset (TEC - SWARM)')
ax[2].set_xlabel('Accuracy')
ax[2].set_title('STD')
ax[3].set_title('TPR')
ax[3].set_xlabel('FPR')
plt.legend()

fig, ax = plt.subplots(1, 3, figsize=(12, 6), tight_layout=True)
fig.suptitle("TV Weight")
lines = []
for v in np.unique(params['l2_weight']):
    mask = params['l2_weight'] == v
    x1 = results['mlon_acc'][mask] - results['mlon_acc'][mask].mean()
    x2 = results['mlon_fpr'][mask] - results['mlon_fpr'][mask].mean()
    y1 = w[mask] - w[mask].mean()
    y2 = s[mask] - s[mask].mean()
    y3 = results['mlon_tpr'][mask] - results['mlon_tpr'][mask].mean()
    for i, k in enumerate(np.unique(params['tv_weight'])):
        mask2 = params['tv_weight'][mask] == k
        ax[0].plot(x1[mask2], y1[mask2], '.', c=cmap[i])
        ax[1].plot(x1[mask2], y2[mask2], '.', c=cmap[i])
        lines += ax[2].plot(x2[mask2], y3[mask2], '.', c=cmap[i], label=k)
ax[0].set_xlabel('accuracy diff')
ax[0].set_title('width diff')
ax[1].set_xlabel('accuracy diff')
ax[1].set_title('std diff')
ax[2].set_xlabel('FPR diff')
ax[2].set_title('TPR diff')
ax[2].legend(handles=lines[-3:])

fig, ax = plt.subplots(1, 3, figsize=(12, 6), tight_layout=True)
fig.suptitle("Background Filter Size")
u = np.unique(params['bg_est_shape'].values)
k = np.argwhere(params['bg_est_shape'].values[:, None] == u[None, :])[:, 1]
lines = []
for v in np.unique(params['l2_weight']):
    mask = params['l2_weight'] == v
    x1 = results['mlon_acc'][mask] - results['mlon_acc'][mask].mean()
    x2 = results['mlon_fpr'][mask] - results['mlon_fpr'][mask].mean()
    y1 = w[mask] - w[mask].mean()
    y2 = s[mask] - s[mask].mean()
    y3 = results['mlon_tpr'][mask] - results['mlon_tpr'][mask].mean()
    for i in range(3):
        mask2 = k[mask] == i
        ax[0].plot(x1[mask2], y1[mask2], '.', c=cmap[i])
        ax[1].plot(x1[mask2], y2[mask2], '.', c=cmap[i])
        lines += ax[2].plot(x2[mask2], y3[mask2], '.', c=cmap[i], label=u[i])
ax[0].set_xlabel('accuracy diff')
ax[0].set_title('width diff')
ax[1].set_xlabel('accuracy diff')
ax[1].set_title('std diff')
ax[2].set_xlabel('FPR diff')
ax[2].set_title('TPR diff')
ax[2].legend(handles=lines[-3:])

fig, ax = plt.subplots(1, 3, figsize=(12, 6), tight_layout=True)
fig.suptitle("Artifact Filter Size")
u = np.unique(params['artifact_key'].values)
k = np.argwhere(params['artifact_key'].values[:, None] == u[None, :])[:, 1]
lines = []
for v in np.unique(params['l2_weight']):
    mask = params['l2_weight'] == v
    x1 = results['mlon_acc'][mask] - results['mlon_acc'][mask].mean()
    x2 = results['mlon_fpr'][mask] - results['mlon_fpr'][mask].mean()
    y1 = w[mask] - w[mask].mean()
    y2 = s[mask] - s[mask].mean()
    y3 = results['mlon_tpr'][mask] - results['mlon_tpr'][mask].mean()
    for i in range(3):
        mask2 = k[mask] == i
        ax[0].plot(x1[mask2], y1[mask2], '.', c=cmap[i])
        ax[1].plot(x1[mask2], y2[mask2], '.', c=cmap[i])
        lines += ax[2].plot(x2[mask2], y3[mask2], '.', c=cmap[i], label=u[i])
ax[0].set_xlabel('accuracy diff')
ax[0].set_title('width diff')
ax[1].set_xlabel('accuracy diff')
ax[1].set_title('std diff')
ax[2].set_xlabel('FPR diff')
ax[2].set_title('TPR diff')
ax[2].legend(handles=lines[-3:])

plt.show()
