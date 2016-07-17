#-*- coding: utf8
from __future__ import division, print_function

import tribeflow
import pandas as pd
import plac
import numpy as np

def main(model, n):
    n = int(n)
    store = pd.HDFStore(model)
    
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    assert from_ == 0
    
    trace_fpath = store['trace_fpath'][0][0]
    kernel_class = store['kernel_class'][0][0]
    kernel_class = eval(kernel_class)

    Theta_zh = store['Theta_zh'].values
    Psi_sz = store['Psi_sz'].values
    count_z = store['count_z'].values[:, 0]
    P = store['P'].values
    residency_priors = store['residency_priors'].values[:, 0]
    
    previous_stamps = StampLists(count_z.shape[0])

    mem_size = store['Dts'].values.shape[1]
    tstamps = store['Dts'].values[:, 0]
    assign = store['assign'].values[:, 0]
    for z in xrange(count_z.shape[0]):
        idx = assign == z
        previous_stamps._extend(z, tstamps[idx])

    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['source2id'].values)
    
    HSDs = []
    Dts = []
    prev = {}
    with open(trace_fpath) as trace_file:
        for i, l in enumerate(trace_file): 
            spl = l.strip().split('\t')
            dts_line = [float(x) for x in spl[:mem_size]]
            h = spl[mem_size]
            d = spl[-1]
            sources = spl[mem_size + 1:-1]
            trace_line = [hyper2id[h]] + [obj2id[s] for s in sources] + \
                    [obj2id[d]]
            if i < to:
                prev[h] = (dts_line, trace_line) 
    
    trace_size = sum(count_z)
    kernel = kernel_class()
    kernel.build(trace_size, count_z.shape[0], residency_priors)
    kernel.update_state(P)
    for i in xrange(n):
        for h in prev:
            max_prob = 0.0
            pred_o = None
            for z in xrange(Psi_sz.shape[1]):
                for candidate_o in xrange(Psi_sz.shape[0]):
                    aux_base[candidate_o] += count_z[z] * mem_factor[z] * \
                        Psi_sz[candidate_o, z] 
            pred_o =
            print(h, pred_o)
            prev[h] = (dts_line[h][:-1] + [pred_dt], prev[h][:-1] + [pred_o])
    
    store.close()
    
plac.call(main)
