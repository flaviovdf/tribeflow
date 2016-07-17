#-*- coding: utf8
from __future__ import division, print_function

import matplotlib.pyplot as plt
import pandas as pd
import plac
import numpy as np
import sys

def main(model_fpath, o_by_o=False, remove_unlikelly=False):
    o_by_o = bool(o_by_o)
    remove_unlikelly = bool(remove_unlikelly)

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
    
    if o_by_o
        T = Psi_zo.dot(Psi_oz)
    else:
        T = Psi_oz.dot(Psi_zo)
    
    np.fill_diagonal(T, 0)
    T = T / T.sum(axis=0)
    
    #h = lambda x: (-x[x != 0] * np.log2(x[x != 0])).sum()
    #for z in xrange(Psi_oz.shape[1]):
    #    print('Z_%d' % z, h(Psi_oz[:, z]) / h(np.ones(Psi_oz.shape[0], dtype='d') /  Psi_oz.shape[0]),\
    #            sep='\t')

    T = T / T.sum(axis=0)

    #for a in xrange(T.shape[0]):
    #    print(id2obj[a], h(T[:, a]) / h(np.ones(len(T), dtype='d') / len(T)), \
    #            sep='\t', file=sys.stderr)
    


    #mean = np.mean(O.ravel())
    #std = np.std(O.ravel())
    #upper = mean + 2 * std
    #lower = mean - 2 * std
    #O[O < upper] = 0
    
    #z_names = ['Z_%d' % i for i in xrange(O.shape[0])]
    #df = pd.DataFrame(data=O, index=z_names, columns=z_names)
    #df.to_csv('ZtoZMatColToRow.dat', sep='\t')

    if o_by_o:
        names = [id2obj[i] for i in xrange(T.shape[0])]
    else:
        names = xrange(Psi_sz.shape[1])

    df = pd.DataFrame(data=T, index=a_names, columns=a_names)
    df.to_csv('mat.dat', sep='\t')
    #plt.imshow(O, cmap=plt.cm.cubehelix_r)
    #plt.colorbar()
    #plt.savefig('tmat-collabs.pdf')

    store.close()
    
plac.call(main)
