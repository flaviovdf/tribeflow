from __future__ import division, print_function

from node_sherlock import _learn

import pandas as pd
import plac
import numpy as np

def main(model, test_file):
    store = pd.HDFStore(model)
    
    mem_size = store['Dts'].values.shape[1]
    Psi_sz = store['Psi_sz'].values
    Theta_zh = store['Theta_zh'].values
    count_z = store['count_z'].values[:, 0]
    
    obj2id = dict(store['source2id'].values)
    hyper2id = dict(store['hyper2id'].values)

    HOs = []
    with open(test_file) as test:
        for l in test: 
            spl = l.strip().split('\t')
            h = spl[mem_size]
            transition = spl[mem_size + 1:]
            
            all_in = h in hyper2id
            for o in transition:
                all_in = all_in and o in obj2id

            if all_in:
                aux = [hyper2id[h]]
                for o in transition:
                    aux.append(obj2id[o])
                HOs.append(aux)
    HOs = np.asarray(HOs, dtype='i4')
   
    a, b = _learn.loglikelihood(HOs, Theta_zh, Psi_sz, count_z)
    print(a, b)
    store.close()
    
plac.call(main)
