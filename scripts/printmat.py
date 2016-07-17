#-*- coding: utf8
from __future__ import division, print_function

import matplotlib.pyplot as plt
import pandas as pd
import plac
import numpy as np
import sys

def main(model_fpath, out_fpath, o_by_o=False):
    o_by_o = bool(o_by_o)
    store = pd.HDFStore(model_fpath)
    
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
    id2obj = dict((v, k) for k, v in obj2id.items())
    
    if o_by_o:
        T = Psi_oz.dot(Psi_zo)
    else:
        T = Psi_zo.dot(Psi_oz)
    
    np.fill_diagonal(T, 0)
    T = T / T.sum(axis=0)
    if o_by_o:
        names = [id2obj[i] for i in xrange(T.shape[0])]
    else:
        names = xrange(T.shape[0])

    df = pd.DataFrame(data=T, index=names, columns=names)
    df.to_csv(out_fpath, sep='\t')
    store.close()
    
plac.call(main)
