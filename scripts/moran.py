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
    parser.add_argument('model_fpath', \
            help='The name of the model file (a h5 file)', type=str)
    parser.add_argument('latlong_fpath', \
            help='The name of the file with lat and long vals', type=str)
    args = parser.parse_args()
    model = pd.HDFStore(args.model_fpath, 'r')         
    
    Psi_sz = model['Psi_sz'].values
    Psi_dz = model['Psi_dz'].values
    count_z = model['count_z'].values
    pz = count_z / count_z.sum()

    Psi_zs = Psi_sz.T * pz
    P_ds = Psi_dz.dot(Psi_zs)
    P_ds = P_ds / P_ds.sum(axis=0)

    dest2id = model['dest2id'].values
    source2id = model['source2id'].values
    
    from collections import Counter
    id2dest = dict((r[1], r[0]) for r in dest2id)
    id2source = dict((r[1], r[0]) for r in source2id)
    
if __name__ == '__main__':
    main()
