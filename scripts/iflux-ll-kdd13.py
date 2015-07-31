import pandas as pd
import plac
import numpy as np
def main(model, test_file):
    store = pd.HDFStore(model)

    Psi_sz = store['Psi_sz'].values
    Psi_dz = store['Psi_dz'].values
    count_z = store['count_z'].values[:, 0]

    T_ds = Psi_dz.dot((Psi_sz * count_z).T) 
    T_ds = T_ds / T_ds.sum(axis=0)
    
    source2id = dict(store['source2id'].values)
    dest2id = dict(store['dest2id'].values)
    
    N_ds = np.zeros_like(T_ds)
    count = {}
    with open(test_file) as test:
        for l in test: 
            _, _, s, d = l.strip().split('\t')
            if s in source2id and d in dest2id:
                N_ds[dest2id[d], source2id[s]] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        L = N_ds * np.log2(T_ds)
        L[np.isnan(L)] = 0
        L[np.isinf(L)] = 0
    
    ll = L.sum()
    print 'total', ll
    print 'avg', ll / N_ds.sum()

    store.close()
    
plac.call(main)
