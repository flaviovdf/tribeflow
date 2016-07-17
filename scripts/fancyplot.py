#-*- coding: utf8
from __future__ import division, print_function

import arrow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plac
import seaborn as sns
import toyplot
import toyplot.pdf
import toyplot.html

def main(model, raw_fpath):
    store = pd.HDFStore(model)
    
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    assert from_ == 0
    
    trace_fpath = store['trace_fpath'][0][0]
    Theta_zh = store['Theta_zh'].values
    Psi_oz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]

    Psi_oz = Psi_oz / Psi_oz.sum(axis=0)
    Psi_zo = (Psi_oz * count_z).T
    Psi_zo = Psi_zo / Psi_zo.sum(axis=0)
    obj2id = dict(store['source2id'].values)
    hyper2id = dict(store['hyper2id'].values)
    id2obj = dict((v, k) for k, v in obj2id.items())

    for name in ['Miles Davis', 'Eric Dolphy', 'Ron Carter', 'Dave Holland', 'John Coltrane']:
        try:
            f = open(raw_fpath)
            series_ts = []
            for l in f:
                ts, c, s, d = l.strip().split('\t')
                ts = float(ts)
                if d == name:
                    series_ts.append([ts, c, s, d])

            PZs = []
            for ts, c, s, d in series_ts:
                PZs.append([])
                for z in xrange(Psi_oz.shape[1]):
                    pzu = Theta_zh[z, hyper2id[c]]
                    psz = Psi_oz[obj2id[s], z]
                    pdz = Psi_oz[obj2id[d], z]
                    pz = (pzu * psz * pdz) / (1 - psz)
                    PZs[-1].append(pz)

            data = []
            index = []
            for i in xrange(len(series_ts)):
                p = PZs[i]
                p = np.asarray(p)
                p = p / p.sum()
                data.append(p)
                index.append(series_ts[i][0])
            heights = []
            x = set()
            for z in xrange(Psi_oz.shape[1]):
                series = pd.Series(data=np.array(data)[:,z], index=pd.DatetimeIndex(np.array(index) * 1e9))
                series = series.resample('%dD'% (2 * 365), how='mean', fill_method='pad')
                heights.append(series.values)
                x.update(series.index)

            x = np.array([arrow.get(d).timestamp for d in sorted(x)])
            heights = np.column_stack(heights)
            canvas = toyplot.Canvas(width=500, height=300)
            axes = canvas.axes(label=name, ylabel='Likelihood (Avg over 2 years)', xlabel='Year')
            color = np.arange(heights.shape[1])
            labels = ['Z_%d' % z for z in xrange(30)]
            to_use = []
            for i in range(heights.shape[1]):
                to_use.append(heights[:, i].max())
            to_use = np.array(to_use)
            print(to_use.argsort()[-5:])
            #heights = heights[:, to_use.argsort()[-5:]]
            #m = axes.plot(x, heights)#, color=color)#, baseline="stacked", color=color)
            m = axes.fill(x, heights, color=color, baseline="stacked")#, color=color)
            label_style = {"text-anchor":"start", "-toyplot-anchor-shift":"0px"}
            #m(0)
            #for i in range(heights.shape[1]):
            #    p = heights[:, i].argmax()
            #    axes.text(x[p], heights[:, i].max(), "Z_%s" % to_use.argsort()[-5:][i], style=label_style, color='black')
        except:
            pass

        #print(m)
        #seen = set()
        #for i, h in enumerate(heights):
        #    top = h.argsort()[::-1]
        #    for t in largest_area:
        #        if t not in seen and len(seen) < 8:
        #            seen.add(t)
        #            axes.text(x[i], h[t], 'Z_%d' % t, \
        #                    style=label_style,
        #                    color='Orange')
        #            break

        #for h in heights:
        #    print(h.argsort()[::-1][:4])
        axes.x.ticks.show = True
        axes.x.ticks.locator = toyplot.locator.Timestamp()
        toyplot.pdf.render(canvas, '%s-stack-2y.pdf' % name)
        toyplot.html.render(canvas, '%s-stack-2y.html' % name)

        f.close()
    #plt.savefig('series.pdf')
    #ZtZ = Psi_zo.dot(Psi_oz)
    #ZtZ = ZtZ / ZtZ.sum(axis=0)
    #L = ZtZ
    #colormap = toyplot.color.brewer.map("Purples", domain_min=L.min(), domain_max=L.max(), reverse=True)
    #canvas = toyplot.matrix((L, colormap), label="P[Z = d | Z = s]", \
    #        colorshow=True)[0]
    #toyplot.pdf.render(canvas, 'test.pdf')
    #AtA = Psi_oz.dot(Psi_zo)
    #np.fill_diagonal(AtA, 0)
    #AtA = AtA / AtA.sum(axis=0)

    store.close()
    
plac.call(main)
