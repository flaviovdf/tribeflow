import pandas as pd
import plac
import numpy as np

from functools32 import lru_cache

def main(model, trace_fpath, last=0):
    last = float(last)

    df = pd.read_csv(trace_fpath, sep='\t', \
            names=['dt1', 'dt2', 'u', 's1', 's2', 'd'], \
            dtype={'dt1':float, 'dt2':float, 'u': str, 's1':str, 's2':str, 'd':str})
    
    use = int(len(df) * (1 - last))
    df = df[use:]

    store = pd.HDFStore(model)

    Psi_dz = store['Psi_sz'].values
    Theta_zh = store['Theta_zh'].values
    count_z = store['count_z'].values[:, 0]

    Psi_dz = Psi_dz / Psi_dz.sum(axis=0)
    Psi_zd = (Psi_dz * count_z).T
    Psi_zd = Psi_zd / Psi_zd.sum(axis=0)

    source2id = dict(store['source2id'].values)
    dest2id = source2id #dict(store['dest2id'].values)
    hyper2id = dict(store['hyper2id'].values)
    
    #@lru_cache(maxsize=100)
    def get_Psi_zs(s):
        aux = Psi_zd[:, source2id[s1]]
        Psi_zs = (Psi_dz * aux).T
        Psi_zs = Psi_zs / Psi_zs.sum(axis=0)
        return Psi_zs

    ll = 0.0
    ll_uni = 0.0
    n = 0
    probs = {}
    
    i = 0
    for _, _, _, s1, s2, d in df.values: 
        if s1 in source2id and s2 in source2id and d in dest2id:
            print(i)
            i += 1

            if (dest2id[d], source2id[s2], source2id[s1]) not in probs:
                Psi_zs = get_Psi_zs(s1)
                probs[dest2id[d], source2id[s2], source2id[s1]] = \
                        (Psi_dz[dest2id[d]] * Psi_zs[:, source2id[s2]]).sum()

            ll += np.log(probs[dest2id[d], source2id[s2], source2id[s1]])
            ll_uni += np.log(1.0 / Psi_dz.shape[0])
            n += 1
            print(ll / n)
    
    print 'total', ll
    print 'avg', ll / n
    print 'avg uni', ll_uni / n

    store.close()
    
plac.call(main)
