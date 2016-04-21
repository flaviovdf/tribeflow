#-*- coding: utf8
from __future__ import division, print_function

from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn import utils

import matplotlib.pyplot as plt
import pandas as pd
import plac
import numpy as np

def main(model):
    store = pd.HDFStore(model)
    
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    assert from_ == 0
    
    trace_fpath = store['trace_fpath'][0][0]
    Psi_oz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]

    Psi_oz = Psi_oz / Psi_oz.sum(axis=0)
    Psi_zo = (Psi_oz * count_z).T
    Psi_zo = Psi_zo / Psi_zo.sum(axis=0)
    obj2id = dict(store['source2id'].values)

    #O = Psi_oz.dot(Psi_zo)
    #O = O / O.sum(axis=0)
    #O = -np.log(O)
    #np.fill_diagonal(O, 0)

    model = TSNE()
    #degrees_of_freedom = max(model.n_components - 1.0, 1)
    #n_samples = O.shape[0]
    #random_state = utils.check_random_state(None)
    #T = model._tsne(squareform(O), degrees_of_freedom, n_samples, random_state)
    T = model.fit_transform(Psi_zo.T)

    plt.plot(T[:, 0], T[:, 1], 'wo')
    plt.savefig('tsne.pdf')

    store.close()
    
plac.call(main)
