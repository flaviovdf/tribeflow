import pandas as pd
import plac
import numpy as np

def main(model, test_file):
    store = pd.HDFStore(model)

    Psi_sz = store['Psi_sz'].values
    Psi_dz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]
    
    Psi_dz = Psi_dz / Psi_dz.sum(axis=0)
    Psi_zs = (Psi_sz * count_z).T
    Psi_zs = Psi_zs / Psi_zs.sum(axis=0)

    #T_ds = Psi_dz.dot(Psi_zs)
    #np.fill_diagonal(T_ds, 0)
    #T_ds = T_ds / T_ds.sum(axis=0)

    source2id = dict(store['source2id'].values)
    dest2id = source2id #dict(store['dest2id'].values)
    
    count_pair = {}
    count_s = {}
    with open(test_file) as test:
        for l in test:
            _, _, s, d = l.strip().split('\t')
            if (d, s) not in count_pair:
                count_pair[d, s] = 0
            if s not in count_s:
                count_s[s] = 0

            count_pair[d, s] += 1.0
            count_s[s] += 1.0
    
    ll = 0.0
    ll_uni = 0.0
    ll_max = 0.0
    n = 0
    probs = {}
    with open(test_file) as test:
        for l in test: 
            _, _, s, d = l.strip().split('\t')
            if s in source2id and d in dest2id:
                if (dest2id[d], source2id[s]) not in probs:
                    probs[dest2id[d], source2id[s]] = \
                            (Psi_dz[dest2id[d]] * Psi_zs[:, source2id[s]]).sum()
                ll += np.log(probs[dest2id[d], source2id[s]]) 
                ll_uni += np.log(1.0 / Psi_dz.shape[0])
                ll_max += np.log(count_pair[d, s] / count_s[s])
                n += 1
    
    print 'total', ll
    print 'avg', ll / n
    print 'avg uni', ll_uni / n
    print ll_max / n

    store.close()
    
plac.call(main)
