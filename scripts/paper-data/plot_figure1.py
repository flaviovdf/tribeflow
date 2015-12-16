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
df = pd.read_excel('results_for_figure1.xlsx').dropna()

styles = {
        'MultiLME':'s', 
        'LDA':'D',
        'TMLDA':'h',
        'TribeFlow':'o'
        }

colors = {
        #('Yes', 50):'g',
        ('Yes', 100):'m',
        ('FourSQ', 10):'b'
        }

for method in styles:
    for dset, k in colors:
        idx_dk = (df['Dataset'] == dset) & (df['K'] == k)
        idx = (df['Name'] == method) & (df['Dataset'] == dset) & (df['K'] == k)
        
        x_ax = df[idx]['Runtime_s']
        y_ax = (df[idx]['LL'] / df[idx_dk]['LL'].max()) ** -1
        
        horizontalalignment = 'left'
        verticalalignment = 'bottom'

        if colors[dset, k] == 'g' and 'LDA' in method:
            horizontalalignment = 'right'
        
        if colors[dset, k] != 'b' and styles[method] == 'D' and 'LDA' in method:
            verticalalignment = 'top'
        
        if colors[dset, k] == 'b' and styles[method] == 'h':
            verticalalignment = 'top'
        
        if 'Flow' in method and colors[dset, k] == 'm':
            verticalalignment = 'top'
            horizontalalignment = 'center'

        for x, y in zip(x_ax, y_ax):
            name = method
            if 'Flow' in method:
                name += '-NT'
            plt.text(x, y, \
                    name + '\n@' + \
                    dset + '', fontsize=7, \
                    verticalalignment=verticalalignment, \
                    horizontalalignment=horizontalalignment)

        ps = colors[dset, k] + styles[method]
        plt.semilogx(df[idx]['Runtime_s'], y_ax, ps, alpha=1, markersize=5)

ax = plt.gca()
ax.tick_params(direction='out', pad=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.ylim((0.6, 1.04))
plt.xlim((30, 1e6 * 0.5))
plt.minorticks_off()
plt.ylabel('Normalized LL (Test Set)', labelpad=0)
plt.xlabel('Execution Time - Wall Clock (s)', labelpad=0)
plt.tight_layout(pad=0.2)
plt.savefig('figure1.pdf')
