#-*- coding: utf8
from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

lme = {50:25755,
       100:68729}

lme_foursq = {10:23049}

iflux = {50:230,
         100:434}

iflux_foursq = {10:51}

print((230.0 / 25755) ** -1, (434.0 / 68729) ** -1, (51.0 / 23049) ** -1)

def initialize_matplotlib():

    inches_per_pt = 1.0 / 72.27
    fig_width = 120 * inches_per_pt  # width in inches
    fig_height = 96 * inches_per_pt  #.4 * fig_width
    
    rc('axes', labelsize=6)
    rc('axes', titlesize=6)
    rc('axes', unicode_minus=False)
    rc('axes', grid=False)
    rc('figure', figsize=(fig_width, fig_height))
    rc('grid', linestyle=':')
    rc('font', family='serif')
    rc('legend', fontsize=5)
    rc('lines', linewidth=.7)
    rc('ps', usedistiller='xpdf')
    rc('text', usetex=True)
    rc('xtick', labelsize=6)
    rc('ytick', labelsize=6)

initialize_matplotlib()

bar_width = 0.35

values_lme = [lme[x] for x in sorted(lme)] + [lme_foursq[x] for x in sorted(lme_foursq)]
values_iflux = [iflux[x] for x in sorted(iflux)] + [iflux_foursq[x] for x in sorted(iflux_foursq)]
index = np.arange(len(values_lme))
rects = plt.bar(index, values_lme, bar_width, alpha=0.3, color='r', label='LME', linewidth=0)
for i, rect in enumerate(rects):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, 'LME', ha='center', va='bottom', fontsize=4)

rects = plt.bar(index + bar_width, values_iflux, bar_width, alpha=0.3, color='b', label='CF', linewidth=0)
for i, rect in enumerate(rects):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, 'CF', ha='center', va='bottom', fontsize=4)

plt.xticks(index + bar_width, (r'Yes 50', r'Yes 100', r'F.Sq 10'))

ax = plt.gca()
ax.set_yscale('log')
ax.set_ylim((10, max(values_lme) * 2))
ax.tick_params(direction='out', pad=0.3)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.minorticks_off()
plt.ylabel('Runtime (s)', labelpad=0)
#plt.legend(frameon=False, loc='upper left')
plt.tight_layout(pad=0.2)
plt.savefig('speed_lme.pdf')
