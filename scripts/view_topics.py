#-*- coding: utf8
from __future__ import division, print_function

from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib
#matplotlib.use('Agg')

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
    Theta_hz = Theta_zh.T * model['count_z'].values[:, 0]
    Theta_hz = Theta_hz / Theta_hz.sum(axis=0)

    Psi_oz = model['Psi_sz'].values
    hyper2id = model['hyper2id'].values
    source2id = model['source2id'].values
    
    from collections import Counter
    counter = Counter(assign)
    
    id2hyper = dict((r[1], r[0]) for r in hyper2id)
    id2source = dict((r[1], r[0]) for r in source2id)
    
    nz = Psi_oz.shape[1]
    k = 10
    print('topic', end='\t')
    for i in xrange(k):
        print('TopUser_%d' %i, 'Pzu_%d' %i, end='\t', sep='\t')
    for i in xrange(k):
        if i != k - 1: 
            print('TopObj_%d' %i, 'Poz_%d' %i, end='\t', sep='\t')
        else:
            print('TopObj_%d' %i, 'Poz_%d' %i, sep='\t')

    for z, pz in counter.most_common()[-nz:]:
        print(z, end='\t')
        #print('These Users\n--')
        for i in Theta_hz[:, z].argsort()[::-1][:k]:
            print(id2hyper[i], Theta_hz[i, z], sep='\t', end='\t')
        #print()

        #print('Transition Through These Objects\n--')
        for i in Psi_oz[:, z].argsort()[::-1][:k]:
            if i != Psi_oz[:, z].argsort()[::-1][k-1]: 
                print(id2source[i], Psi_oz[i, z], sep='\t', end='\t')
            else:
                print(id2source[i], Psi_oz[i, z], sep='\t')
        #print()
        #print()

    model.close()

if __name__ == '__main__':
    main()
