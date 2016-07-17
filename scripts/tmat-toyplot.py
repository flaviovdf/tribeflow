#-*- coding: utf8
from __future__ import division, print_function

from sklearn.cluster.bicluster import SpectralCoclustering

import arrow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plac
import seaborn as sns
import toyplot
import toyplot.pdf
import toyplot.html

def main(model):
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

    ZtZ = Psi_zo.dot(Psi_oz)
    ZtZ = ZtZ / ZtZ.sum(axis=0)
    L = ZtZ
    #ZtZ[ZtZ < (ZtZ.mean())] = 0
    L[ZtZ >= 1.0 / (len(ZtZ))] = 1
    L[L != 1] = 0

    colormap = toyplot.color.brewer.map("Purples", domain_min=0, domain_max=1, reverse=True)
    print(colormap)
    canvas = toyplot.matrix((L.T, colormap), label="P[z' | z]", \
            colorshow=False, tlabel="To z'", llabel="From")[0]
    #canvas.axes(ylabel='From z', xlabel='To z\'')
    toyplot.pdf.render(canvas, 'tmat.pdf')

    model = SpectralCoclustering(n_clusters=3)
    model.fit(L)
    fit_data = L[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    canvas = toyplot.matrix((fit_data, colormap), label="P[z' | z']", \
            colorshow=False)[0]
    toyplot.pdf.render(canvas, 'tmat-cluster.pdf')
    
    #AtA = Psi_oz.dot(Psi_zo)
    #np.fill_diagonal(AtA, 0)
    #AtA = AtA / AtA.sum(axis=0)

    store.close()
    
plac.call(main)
