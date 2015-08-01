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
    deltas = model['tstamps'].values[:, 0]
    Psi_sz = model['Psi_sz'].values
    Psi_dz = model['Psi_dz'].values
    dest2id = model['dest2id'].values
    source2id = model['source2id'].values
    
    print(Psi_sz)
    print(Psi_dz)

    from collections import Counter
    counter = Counter(assign)
    print(len(counter))
    print(counter)
    print()

    id2dest = dict((r[1], r[0]) for r in dest2id)
    id2source = dict((r[1], r[0]) for r in source2id)
    
    nz = 20 #Psi_sz.shape[1]
    rows = int(np.ceil(nz / 5))
    cols = 5
    k = 10
    p = 0
    for z, pz in counter.most_common()[-nz:]:
        plt.subplot(rows, cols, p + 1)
        p += 1

        deltas_z = deltas[assign == z]
        deltas_z = deltas_z[deltas_z > 0]
        #deltas_z = deltas_z[deltas_z < 5000]

        ecdf = ECDF(deltas_z)
        #x = np.linspace(deltas_z.min(), deltas_z.max(), 1000)
        x = np.unique(deltas_z)
        print(x)
        #plt.hist(deltas_z, bins=24*12)
        #plt.loglog(x, 1 - ecdf(x))
        plt.plot(x, 1 - ecdf(x))
        plt.title(deltas_z.mean())
        top_source = np.argpartition(-Psi_sz[:, z], k)
        top_dest = np.argpartition(-Psi_dz[:, z], k)
        print(Psi_sz[:, z].sum()) 
        print(z)
        for i in xrange(k):
            print(id2source[top_source[i]])
        print()

        for i in xrange(k):
            print(id2dest[top_dest[i]])
        print()
        print()
        print()

    plt.tight_layout(pad=3)
    #plt.savefig('view.pdf')
    plt.show()
    model.close()

if __name__ == '__main__':
    main()
