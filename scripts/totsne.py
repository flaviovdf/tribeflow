#-*- coding: utf8
from __future__ import division, print_function

from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib
#matplotlib.use('Agg')
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_fpath', help='The name of the model file (a h5 file)', \
            type=str)
    args = parser.parse_args()
    model = pd.HDFStore(args.model_fpath, 'r')         
    
    assign = model['assign'].values[:, 0]
    Theta_zh = model['Theta_zh'].values
    hyper2id = model['hyper2id'].values
    id2hyper = dict((r[1], r[0]) for r in hyper2id)
    
    Psi_oz = model['Psi_sz'].values
    count_z = model['count_z'].values[:, 0]

    Psi_oz = Psi_oz / Psi_oz.sum(axis=0)
    Psi_zo = (Psi_oz * count_z).T
    Psi_zo = Psi_zo / Psi_zo.sum(axis=0)
    obj2id = dict(model['source2id'].values)
    id2obj = dict((r[1], r[0]) for r in obj2id.items())
    from collections import Counter
    counter = Counter(assign)
    
    print(end='\t')
    print('\t'.join('Z_%d' % i for i in xrange(Theta_zh.shape[0])))
    for h in xrange(Theta_zh.shape[1]):
        p = '\t'.join(['%.32f' % x for x in  Theta_zh[:, h]])
        print(id2hyper[h], p, sep='\t')
    
    print(end='\t', file=sys.stderr)
    print('\t'.join(id2obj[i] for i in xrange(Psi_oz.shape[0])), \
            file=sys.stderr)
    i = 0
    for o in xrange(Psi_oz.shape[1]):
        p = '\t'.join(['%.32f' % x for x in  Psi_oz[o]])
        print('Z_%d' % i, p, sep='\t', file=sys.stderr)
        i += 1
    model.close()

if __name__ == '__main__':
    main()
