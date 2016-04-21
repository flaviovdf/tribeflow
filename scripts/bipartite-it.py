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

    Psi_oz = model['Psi_sz'].values
    hyper2id = model['hyper2id'].values
    source2id = model['source2id'].values
    
    from collections import Counter
    
    id2hyper = dict((r[1], r[0]) for r in hyper2id)
    id2source = dict((r[1], r[0]) for r in source2id)
    
    nz = Psi_oz.shape[1]
    k = 10
    for z in xrange(nz): 
        print(z)
        print('These Contributors (name, P[z|c])\n--')
        n = len(Theta_zh[z])
        p = 1.0 / n
        t = p + 1.96 * np.sqrt((1.0 / n) * p * (1 - p))

        for i in Theta_zh[z].argsort()[::-1][:k]:
            if Theta_zh[z, i] > t:
                print(id2hyper[i], Theta_zh[z, i], sep='\t')
        print()

        print('Transition Through These Artists (name, P[a|z])\n--')
        n = len(Psi_oz[:, z])
        p = 1.0 / n
        t = p + 1.96 * np.sqrt((1.0 / n) * p * (1 - p))
        for i in Psi_oz[:, z].argsort()[::-1][:k]:
            if Psi_oz[i, z] > t:
                print(id2source[i], Psi_oz[i, z], sep='\t')
        print()
        print()

    model.close()

if __name__ == '__main__':
    main()
