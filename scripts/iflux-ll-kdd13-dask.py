#-*- coding: utf8
from __future__ import division, print_function

import dask.array as da
import h5py
import tempfile
import pandas as pd
import plac
import numpy as np
import os

def to_dask_from_array(name, X, matrix_file):
    dset = matrix_file.create_dataset(name, X.shape, X.dtype, data=X)
    X_dask = da.from_array(dset, chunks=10000)
    return X_dask

def empty_dask(name, shape, dtype, matrix_file):
    dset = matrix_file.create_dataset(name, shape, dtype, fillvalue=0.0)
    return da.from_array(dset, chunks=(10000, 10000))

def main(model, test_file, tmp_folder_prefix=None):
    store = pd.HDFStore(model)

    if tmp_folder_prefix:
        tmp_folder = tempfile.mkdtemp(prefix=tmp_folder_prefix)
    else:
        tmp_folder = tempfile.mkdtemp()
    
    matrix_fpath = os.path.join(tmp_folder, 'matrices.h5')
    matrix_file = h5py.File(matrix_fpath)

    mem_size = store['Dts'].values.shape[1]
    Theta_zh = store['Theta_zh'].values
    Psi_sz = store['Psi_sz'].values
    Psi_dz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]
    
    Psi_dz = Psi_dz / Psi_dz.sum(axis=0)
    Psi_zs = (Psi_sz * count_z).T
    Psi_zs = Psi_zs / Psi_zs.sum(axis=0)
    Theta_zh = Theta_zh / Theta_zh.sum(axis=0)
    
    Psi_dz = to_dask_from_array('Psi_dz', Psi_dz, matrix_file)
    Psi_zs = to_dask_from_array('Psi_zs', Psi_zs, matrix_file)
    Theta_zh = to_dask_from_array('Theta_zh', Theta_zh, matrix_file)
    
    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['source2id'].values)
    
    probs_personalized = {}
    probs = {}

    ll = 0.0
    ll_uni = 0.0
    n = 0.0
    with open(test_file) as test:
        for l in test: 
            spl = l.strip().split('\t')
            
            h = spl[mem_size]
            memory = tuple(spl[mem_size + 1:-1])
            destination = spl[-1]

            all_in = h in hyper2id and destination in obj2id 
            for o in memory:
                all_in = all_in and o in obj2id
            
            if all_in:
                h = hyper2id[h]
                destination = obj2id[destination]
                memory = np.array([obj2id[o] for o in memory])

                transition_non_personalized = (destination, ) + tuple(memory)
                transition_personalized = (destination, h) + tuple(memory)

                if transition_non_personalized not in probs:
                    if memory.shape[0] > 1:
                        probs[transition_non_personalized] = \
                                (Psi_zs[:, memory[0]] * \
                                 Psi_dz[destination] * \
                                 Psi_dz[memory[1:]].multiply(axis=1)).sum()
                    else:
                        probs[transition_non_personalized] = \
                                (Psi_zs[:, memory[0]] * \
                                 Psi_dz[destination]).sum()
                #if transition_personalized not in probs_personalized:
                #    nrm_cte[transition_personalized[1:]] = \
                #            Psi_dz.dot((Theta_zh[:, h] * Psi_dz).T)
                #    probs_personalized[transition_personalized] = \
                #            (Theta_zh[:, h] * Psi_dz[memory] * Psi_dz[destination]).sum()

                ll += np.log(probs[transition_non_personalized])
                ll_uni += np.log(1.0 / Psi_dz.shape[0])
                n += 1
    
    print(ll, ll / n)
    import shutil
    shutil.rmtree(tmp_folder, ignore_errors=True)
    store.close()
    
plac.call(main)
