import pandas as pd
import plac
import numpy as np

def main(model, trace_fpath, leaveout=0.3):
    leaveout = float(leaveout)
    
    df = pd.read_csv(trace_fpath, sep='\t', names=['dt', 'u', 's', 'd'], \
            dtype={'dt':float, 'u': str, 's':str, 'd':str})
    num_lines = len(df)
    to = int(num_lines - num_lines * leaveout)

    df_train = df[:to]
    df_test = df[to:]

    store = pd.HDFStore(model)

    Psi_dz = store['Psi_sz'].values
    Theta_zh = store['Theta_zh'].values
    count_z = store['count_z'].values[:, 0]

    Psi_dz = Psi_dz / Psi_dz.sum(axis=0)
    Theta_zh = Theta_zh / Theta_zh.sum(axis=0)

    source2id = dict(store['source2id'].values)
    dest2id = source2id #dict(store['dest2id'].values)
    hyper2id = dict(store['hyper2id'].values)

    ll = 0.0
    ll_uni = 0.0
    n = 0
    probs = {}
    
    for _, h, s, d in df_test.values: 
        if h in hyper2id and s in source2id and d in dest2id:
            aux = Theta_zh[:, hyper2id[h]]
            Psi_zs = (Psi_dz * aux).T
            Psi_zs = Psi_zs / Psi_zs.sum(axis=0)
            
            if (hyper2id[h], dest2id[d], source2id[s]) not in probs:
                probs[hyper2id[h], dest2id[d], source2id[s]] = \
                        (Psi_dz[dest2id[d]] * Psi_zs[:, source2id[s]]).sum()
            ll += np.log(probs[hyper2id[h], dest2id[d], source2id[s]])
            ll_uni += np.log(1.0 / Psi_dz.shape[0])
            n += 1
    
    print 'total', ll
    print 'avg', ll / n
    print 'avg uni', ll_uni / n

    store.close()
    
plac.call(main)
