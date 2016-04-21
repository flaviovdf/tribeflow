#-*- coding: utf8
from __future__ import division, print_function

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

    obj2id = dict(store['source2id'].values)

    Psi_oz = Psi_oz / Psi_oz.sum(axis=0)
    Psi_zo = (Psi_oz * count_z).T
    Psi_zo = Psi_zo / Psi_zo.sum(axis=0)

    mem_size = store['Dts'].values.shape[1]
    
    probs = {}
    ll = 0.0
    n = 0.0
    with open(trace_fpath) as trace_file:
        for i, l in enumerate(trace_file): 
            if i >= to:
                break
            
            n += 1
            spl = l.strip().split('\t')
            _, _, s, d = spl
            if (obj2id[d], obj2id[s]) not in probs:
                probs[obj2id[d], obj2id[s]] = \
                        (Psi_oz[obj2id[d]] * Psi_zo[:, obj2id[s]]).sum()
            ll += np.log(probs[obj2id[d], obj2id[s]])

    print(-2 * ll + n * np.log(count_z.shape[0] + sum(Psi_oz.shape)))
    store.close()
    
plac.call(main)
