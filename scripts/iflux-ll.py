import pandas as pd
import plac
import numpy as np

def main(model):
    store = pd.HDFStore(model)
    
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    assert from_ == 0
    
    trace_fpath = store['trace_fpath'][0][0]
    Psi_sz = store['Psi_sz'].values
    Psi_dz = store['Psi_dz'].values
    count_z = store['count_z'].values[:, 0]
    
    #T_ds = Psi_dz.dot((Psi_sz * count_z).T) 
    #T_ds = T_ds / T_ds.sum(axis=0)
    
    source2id = dict(store['source2id'].values)
    dest2id = dict(store['dest2id'].values)
    
    Psi_dz = Psi_dz / Psi_dz.sum(axis=0)
    Psi_zs = (Psi_sz * count_z).T
    Psi_zs = Psi_zs / Psi_zs.sum(axis=0)

    probs = {}
    n = 0.0
    ll = 0.0
    ll_uni = 0.0
    with open(trace_fpath) as trace_file:
        for i, l in enumerate(trace_file):
            if i < to:
                continue
            
            _, _, s, d = l.strip().split('\t')
            if s in source2id and d in dest2id:
                if (dest2id[d], source2id[s]) not in probs:
                    probs[dest2id[d], source2id[s]] = \
                            (Psi_dz[dest2id[d]] * Psi_zs[:, source2id[s]]).sum()
                ll += np.log(probs[dest2id[d], source2id[s]])
                ll_uni += 1.0 / Psi_dz.shape[0]
                n += 1
    
    print 'total', ll
    print 'avg', ll / n
    print 'uni', ll / n

    store.close()
    
plac.call(main)
