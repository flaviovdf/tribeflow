#-*- coding: utf8
from __future__ import division, print_function

from node_sherlock import StampLists

from collections import defaultdict
from collections import OrderedDict

import numpy as np
import pandas as pd

def save_model(out_fpath, model):
    store = pd.HDFStore(out_fpath, 'w')
    for model_key in model:
        model_val = model[model_key]
        
        if type(model_val) == np.ndarray:
            store[model_key] = pd.DataFrame(model_val)
        else:
            store[model_key] = pd.DataFrame(model_val.items(), \
                    columns=['Name', 'Id'])
    store.close()

def initialize_trace(trace_fpath, num_topics, num_iter, \
        from_=0, to=np.inf, initial_assign=None):
    
    count_zh_dict = defaultdict(int)
    count_sz_dict = defaultdict(int)
    count_dz_dict = defaultdict(int)
    count_z_dict = defaultdict(int)
    count_h_dict = defaultdict(int)

    hyper2id = OrderedDict()
    source2id = OrderedDict()
    dest2id = OrderedDict()
    
    topic_stamps_dict = defaultdict(list)
    if initial_assign:
        initial_assign = np.asarray(initial_assign, dtype='i')
        assert initial_assign.min() >= 0
        assert initial_assign.max() < num_topics

    tstamps = []
    Trace = []
    with open(trace_fpath, 'r') as trace_file:
        for i, line in enumerate(trace_file):
            if i < from_ or i >= to:
                continue
            
            tstamp, hyper_str, source_str, dest_str = line.strip().split('\t')
            tstamp = float(tstamp)

            if hyper_str not in hyper2id:
                hyper2id[hyper_str] = len(hyper2id)
            
            if source_str not in source2id:
                source2id[source_str] = len(source2id)
            
            if dest_str not in dest2id:
                dest2id[dest_str] = len(dest2id)

            h = hyper2id[hyper_str]
            s = source2id[source_str]
            d = dest2id[dest_str]
            
            if not initial_assign:
                z = np.random.randint(num_topics)
            else:
                z = initial_assign[i]

            count_zh_dict[z, h] += 1
            count_sz_dict[s, z] += 1
            count_dz_dict[d, z] += 1
            count_z_dict[z] += 1
            count_h_dict[h] += 1
            
            topic_stamps_dict[z].append(i)
            tstamps.append(tstamp)
            Trace.append([h, s, d, z])
    
    #Sort the time stamps 
    tstamps = np.asarray(tstamps)
    argsort = tstamps.argsort()
    tstamps = tstamps[argsort]
    
    #Create contiguous arrays, not needed but adds a small speedup
    tstamps = np.asanyarray(tstamps, order='C')
    Trace = np.asarray(Trace)
    Trace = np.asanyarray(Trace[argsort], dtype='i4', order='C')
    
    nh = len(hyper2id)
    ns = len(source2id)
    nd = len(dest2id)
    nz = num_topics

    previous_stamps = StampLists(num_topics)
    for z in xrange(nz):
        idx = Trace[:, -1] == z
        topic_stamps = tstamps[idx]
        previous_stamps._extend(z, topic_stamps)

    Count_zh = np.zeros(shape=(nz, nh), dtype='i4')
    Count_sz = np.zeros(shape=(ns, nz), dtype='i4')
    Count_dz = np.zeros(shape=(nd, nz), dtype='i4')
    count_h = np.zeros(shape=(nh,), dtype='i4')
    count_z = np.zeros(shape=(nz,), dtype='i4')

    for z in xrange(Count_zh.shape[0]):
        count_z[z] = count_z_dict[z]

        for h in xrange(Count_zh.shape[1]):
            count_h[h] = count_h_dict[h]
            Count_zh[z, h] = count_zh_dict[z, h]

        for s in xrange(Count_sz.shape[0]):
            Count_sz[s, z] = count_sz_dict[s, z]

        for d in xrange(Count_dz.shape[0]):
            Count_dz[d, z] = count_dz_dict[d, z]
    
    prob_topics_aux = np.zeros(nz, dtype='f8')
    
    Theta_zh = np.zeros(shape=(nz, nh), dtype='f8')
    Psi_sz = np.zeros(shape=(ns, nz), dtype='f8')
    Psi_dz = np.zeros(shape=(nd, nz), dtype='f8')

    return tstamps, Trace, previous_stamps, Count_zh, Count_sz, \
            Count_dz, count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id
