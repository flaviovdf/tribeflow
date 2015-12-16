#-*- coding: utf8
from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')

from matplotlib import rc

import matplotlib.pyplot as plt
import pandas as pd

def initialize_matplotlib():
    inches_per_pt = 1.0 / 72.27
    fig_width = 240 * inches_per_pt  # width in inches
    fig_height = 160 * inches_per_pt  #.4 * fig_width
    
    rc('axes', labelsize=8)
    rc('axes', titlesize=8)
    rc('axes', unicode_minus=False)
    rc('axes', grid=False)
    rc('figure', figsize=(fig_width, fig_height))
    rc('grid', linestyle=':')
    rc('font', family='serif')
    rc('legend', fontsize=8)
    rc('lines', linewidth=.7)
    rc('ps', usedistiller='xpdf')
    rc('text', usetex=True)
    rc('xtick', labelsize=8)
    rc('ytick', labelsize=8)

initialize_matplotlib()
df = pd.read_excel('results_for_figure1.xlsx', sheetname='Figure2')

styles = {
        'PRLME':'s', 
        'FPMC':'D',
        'TribeFlow':'o'
        }

colors = {
        'LFM-1k':'g',
        'LFM-G':'m',
        'Bkite':'y',
        'FourSQ':'b',
        'Yoo':'r'
        }

for method in styles:
    for dset in colors:
        idx = (df['Name'] == method) & (df['Dataset'] == dset)
        
        x_ax = df[idx]['Runtime_s']
        y_ax = df[idx]['MRR'] 
        
        horizontalalignment = 'left'
        verticalalignment = 'bottom'
        
        if method == 'TribeFlow' and colors[dset] == 'g':
            verticalalignment = 'top'

        for x, y in zip(x_ax, y_ax):
            if method == 'FPMC' or (method == 'PRLME' and dset != 'Bkite'):
                continue

            plt.text(x, y, \
                    method + ' @ ' + \
                    dset, fontsize=8, \
                    verticalalignment=verticalalignment, \
                    horizontalalignment=horizontalalignment)

        ps = colors[dset] + styles[method]
        plt.semilogx(x_ax, y_ax, ps, alpha=1, markersize=5)

ax = plt.gca()
ax.tick_params(direction='out', pad=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.ylim((-.05, 0.7))
plt.xlim((1e2, 1e5))
plt.minorticks_off()
plt.ylabel('Mean Reciprocal Rank (MRR)', labelpad=0)
plt.xlabel('Execution Time (s)', labelpad=0)
plt.tight_layout(pad=0.2)
plt.savefig('figure2.pdf')
