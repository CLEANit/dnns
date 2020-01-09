#!/usr/bin/env python

import sys 
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cm
from matplotlib import rc
rc('font',**{'family':'sans-serif'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)

def expRunningAvg(current_data, alpha=0.95):
    avgs = [current_data[0]]
    for i, elem in enumerate(current_data[1:]):
        avgs.append(avgs[i] * alpha + (1 - alpha) * elem)
    return avgs

data = np.loadtxt(sys.argv[1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(data[:, 0], data[:, 1], lw=3, c=cm.thermal_r(0.5), alpha=0.2, label='Training loss')
axes[0].plot(data[:, 0], expRunningAvg(data[:, 1]), lw=3, c=cm.thermal_r(0.5), label='Averaged Training loss')
axes[0].plot(data[:, 0], data[:, 2], lw=3, c=cm.thermal_r(0.75), alpha=0.2, label='Validation loss')
axes[0].plot(data[:, 0], expRunningAvg(data[:, 2]), lw=3, c=cm.thermal_r(0.75), label='Averaged Validation loss')

axes[0].set_xlabel('Number of Epochs')
axes[0].set_ylabel('Mean Squared Error')
axes[0].grid(color=cm.thermal_r(0.95), linestyle='-.')
axes[0].legend()
axes[0].set_ylim([0.0, 1.5* np.mean(data[:,1])])
# axes[0].set_facecolor(cm.thermal_r(0.05))

axes[1].plot(np.log10(data[:, 0]), np.log10(data[:, 1]), lw=3, c=cm.thermal_r(0.5), label='Training loss')
axes[1].plot(np.log10(data[:, 0]), np.log10(data[:, 2]), lw=3, c=cm.thermal_r(0.75), label='Validation loss')
axes[1].set_xlabel('log(Number of Epochs)')
axes[1].set_ylabel('log(Mean Squared Error)')
axes[1].grid(color=cm.thermal_r(0.95), linestyle='-.')
axes[1].legend()
# axes[1].set_facecolor(cm.haline_r(0.05))


plt.tight_layout()
plt.show()
