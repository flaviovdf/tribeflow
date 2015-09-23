#-*- coding: utf8
from __future__ import division, print_function

from node_sherlock.mycollections.stamp_lists import StampLists

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
    count_oz_dict = defaultdict(int)
    count_z_dict = defaultdict(int)
    count_h_dict = defaultdict(int)

    hyper2id = OrderedDict()
    obj2id = OrderedDict()
    
    if initial_assign:
        initial_assign = np.asarray(initial_assign, dtype='i')
        assert initial_assign.min() >= 0
        assert initial_assign.max() < num_topics

    Dts = []
    Trace = []
    with open(trace_fpath, 'r') as trace_file:
        for i, line in enumerate(trace_file):
            if i < from_: 
                continue

            if i >= to:
                break

            spl = line.strip().split('\t')
            assert len(spl) >= 4
            assert (len(spl) - 2) % 2 == 0
            mem_size = (len(spl) - 2) // 2
            
            line_dts = []
            for j in xrange(mem_size):
                line_dts.append(float(spl[j]))
            Dts.append(line_dts)

            hyper_str = spl[mem_size]
            if hyper_str not in hyper2id:
                hyper2id[hyper_str] = len(hyper2id)
            
            if not initial_assign:
                z = np.random.randint(num_topics)
            else:
                z = initial_assign[i]

            h = hyper2id[hyper_str]
            count_zh_dict[z, h] += 1
            count_h_dict[h] += 1
            
            line_int = [h]
            for j in xrange(mem_size + 1, len(spl)):
                obj_str = spl[j]
                
                if obj_str not in obj2id:
                    obj2id[obj_str] = len(obj2id)
            
                o = obj2id[obj_str]
                line_int.append(o)
                
                count_oz_dict[o, z] += 1
                count_z_dict[z] += 1 
                       
            line_int.append(z)
            Trace.append(line_int)
    
    #Sort by the last residency time. 
    Dts = np.asarray(Dts)
    argsort = Dts[:, -1].argsort()
    assert Dts.shape[1] == mem_size

    #Create contiguous arrays, not needed but adds a small speedup
    Dts = np.asanyarray(Dts[argsort], order='C')
    Trace = np.asarray(Trace)
    Trace = np.asanyarray(Trace[argsort], dtype='i4', order='C')

    nh = len(hyper2id)
    no = len(obj2id)
    nz = num_topics
    
    previous_stamps = StampLists(num_topics)
    for z in xrange(nz):
        idx = Trace[:, -1] == z
        topic_stamps = Dts[:, -1][idx]
        previous_stamps._extend(z, topic_stamps)

    Count_zh = np.zeros(shape=(nz, nh), dtype='i4')
    Count_oz = np.zeros(shape=(no, nz), dtype='i4')
    count_h = np.zeros(shape=(nh,), dtype='i4')
    count_z = np.zeros(shape=(nz,), dtype='i4')
    
    for z in xrange(Count_zh.shape[0]):
        count_z[z] = count_z_dict[z]

        for h in xrange(Count_zh.shape[1]):
            count_h[h] = count_h_dict[h]
            Count_zh[z, h] = count_zh_dict[z, h]

        for o in xrange(Count_oz.shape[0]):
            Count_oz[o, z] = count_oz_dict[o, z]
    
    assert (Count_oz.sum(axis=0) == count_z).all()

    prob_topics_aux = np.zeros(nz, dtype='f8')
    Theta_zh = np.zeros(shape=(nz, nh), dtype='f8')
    Psi_oz = np.zeros(shape=(no, nz), dtype='f8')
    
    return Dts, Trace, previous_stamps, Count_zh, Count_oz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_oz, \
            hyper2id, obj2id
