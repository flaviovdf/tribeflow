#-*- coding: utf8
'''
This module contains the distributed learning approach we employ
on node_sherlock. As of the moment, parallelization is only performed
on a single machine. 
'''
#TODO: This code is functional but needs cleaning up
#TODO: There is a large overhad in sending arrays using multiprocessing
#      I need to look into a sollution using mpi4py.

from __future__ import division, print_function

from _learn import _aggregate
from _learn import em
from _learn import fast_populate

from collections import OrderedDict

from learn import prepare_results

from node_sherlock import dataio
from node_sherlock import kernels
from node_sherlock import StampLists

import multiprocessing as mp
import numpy as np
import time

def find_a_pair(pairs_lock, available_to_pair, pairs, thread_id):
    with pairs_lock:
        for i in xrange(len(available_to_pair)):
            if available_to_pair[i]:
                available_to_pair[i] = False
                available_to_pair[thread_id] = False

                pairs[i] = thread_id
                pairs[thread_id] = i
                return True, i

        available_to_pair[thread_id] = True
        return False, -1

def paired_update(pairs_lock, available_to_pair, pairs, conditions, \
        Count_sz_others, Count_dz_others, \
        all_Count_szs, all_Count_dzs, all_Ps, P, \
        previous_encounters_s, previous_encounters_d, thread_id):
    
    with conditions[thread_id]:
        paired, my_pair = find_a_pair(pairs_lock, available_to_pair, pairs, \
                thread_id)

        if paired: 
            #Get copies from manager
            P_pair = all_Ps[my_pair]
            Count_sz_pair = all_Count_szs[my_pair]
            Count_dz_pair = all_Count_dzs[my_pair]
        
            #Notify and wait until my pair has data it needs
            with conditions[my_pair]:
                conditions[my_pair].notify()
            conditions[thread_id].wait()
        else: 
            #wait until someone matches with me and get id
            conditions[thread_id].wait()
            my_pair = pairs[thread_id]
            
            #Get copies from manager
            P_pair = all_Ps[my_pair]
            Count_sz_pair = all_Count_szs[my_pair]
            Count_dz_pair = all_Count_dzs[my_pair]
        
            #Notify pair that everything is here
            with conditions[my_pair]:
                conditions[my_pair].notify()
     
    #Update Counts
    N_til_s = previous_encounters_s[my_pair]
    N_til_d = previous_encounters_d[my_pair]
    
    #[:] is to avoid copies of arrays. Make sure we dont lose anything
    Count_sz_others[:] = Count_sz_others + Count_sz_pair - N_til_s
    Count_dz_others[:] = Count_dz_others + Count_dz_pair - N_til_d

    N_til_s[:] = Count_sz_pair
    N_til_d[:] = Count_dz_pair
    P[:] = (P + P_pair) / 2.0

def run_one(tstamps, Trace, all_Ps, all_Count_zhs, all_Count_szs, \
        all_Count_dzs, all_count_hs, all_count_zs, all_assign, \
        alpha_zh, beta_zs, beta_zd, kernel_class, residency_priors, \
        Theta_zh, Psi_sz, Psi_dz, pairs_lock, available_to_pair, pairs, \
        conditions, previous_encounters_s, previous_encounters_d, \
        working_lock, started, ended, thread_id, num_iter):
    
    t = thread_id
    with working_lock:
        started[t] = True
    
    Count_zh = all_Count_zhs[t]
    Count_sz_local = all_Count_szs[t]
    Count_dz_local = all_Count_dzs[t]
    count_h = all_count_hs[t]
    count_z = all_count_zs[t]
    
    assert not Count_zh.any()
    assert not Count_sz_local.any()
    assert not Count_dz_local.any()
    assert not count_h.any()
    assert not count_z.any()
    
    fast_populate(Trace, Count_zh, Count_sz_local, Count_dz_local, count_h, \
            count_z)

    kernel = kernel_class()
    kernel.build(Trace.shape[0], Count_zh.shape[0], residency_priors)
    kernel.update_state(all_Ps[t])

    Count_sz_others = np.zeros_like(Count_sz_local)
    Count_dz_others = np.zeros_like(Count_dz_local)

    Count_sz_sum = np.zeros_like(Count_sz_local)
    Count_dz_sum = np.zeros_like(Count_dz_local)

    #any number larger than 1 will do, used to ignore the burn in step
    #since we perform one iter at a time
    burn_in = 2
    stamps = StampLists(Count_zh.shape[0])
    for z in xrange(Count_zh.shape[1]):
        idx = Trace[:, -1] == z
        stamps._extend(z, tstamps[idx])

    aux = np.zeros(Count_zh.shape[0], dtype='f8')
    
    for i in xrange(num_iter):
        #Sample from the local counts and encountered counts
        Count_sz_sum[:] = Count_sz_local + Count_sz_others
        Count_dz_sum[:] = Count_dz_local + Count_dz_others
        
        em(tstamps, Trace, stamps, Count_zh, Count_sz_sum, \
                Count_dz_sum, count_h, count_z, alpha_zh, beta_zs, \
                beta_zd, aux, Theta_zh, Psi_sz, Psi_dz, 1, burn_in, \
                kernel)

        #Update local counts
        Count_sz_local[:] = Count_sz_sum - Count_sz_others
        Count_dz_local[:] = Count_dz_sum - Count_dz_others
        
        #Do I have a chance at pairing?
        if sum(ended) != sum(ended) - 1:
            #Send copies to manager
            all_Count_szs[t] = Count_sz_local
            all_Count_dzs[t] = Count_dz_local
        
            #Update expected belief of other processors
            P = kernel.get_state()
            paired_update(pairs_lock, available_to_pair, pairs, conditions, \
                    Count_sz_others, Count_dz_others, all_Count_szs, \
                    all_Count_dzs, all_Ps, P, previous_encounters_s, \
                    previous_encounters_d, thread_id)
            kernel.update_state(P)

        #if (i + 1) % 100 == 0:
        #    print('Processor %d is at iter %d' % (t, i))

    #Update manager with final state
    all_Ps[t] = kernel.get_state().copy()
    all_Count_zhs[t] = Count_zh.copy()
    all_Count_szs[t] = Count_sz_local.copy()
    all_Count_dzs[t] = Count_dz_local.copy()
    all_count_hs[t] = count_h.copy()
    all_count_zs[t] = count_z.copy()
    all_assign[t] = Trace[:, -1].copy()

    with working_lock:
        ended[t] = True

def parallel_fit(tstamps, Trace, previous_stamps, Count_zh, Count_sz, \
        Count_dz, count_h, count_z, alpha_zh, beta_zs, beta_zd, kernel, \
        residency_priors, prob_topics_aux, Theta_zh, Psi_sz, Psi_dz, \
        num_iter, workloads, num_threads):
       
    manager = mp.Manager()
    
    all_Ps = manager.list()
    all_Count_zhs = manager.list()
    all_Count_szs = manager.list()
    all_Count_dzs = manager.list()
    all_count_hs = manager.list()
    all_count_zs = manager.list()
    all_assign = manager.list()
    
    threads = []
    conditions = []

    started = manager.list()
    ended = manager.list()
    pairs = manager.list()
    available_to_pair = manager.list()
    pairs_lock = manager.RLock()
    working_lock = manager.RLock()

    for t in xrange(num_threads):
        started.append(False)
        ended.append(False)

        conditions.append(manager.Condition())
        available_to_pair.append(False)
        pairs.append(-1)
        
        all_Ps.append(kernel.get_state())
        all_Count_zhs.append(np.zeros_like(Count_zh))
        all_Count_szs.append(np.zeros_like(Count_sz))
        all_Count_dzs.append(np.zeros_like(Count_dz))
        all_count_hs.append(np.zeros_like(count_h))
        all_count_zs.append(np.zeros_like(count_z))
        all_assign.append(0)

        previous_encounters_s = []
        previous_encounters_d = []
        for o in xrange(num_threads):
            previous_encounters_s.append(np.zeros_like(Count_sz))
            previous_encounters_d.append(np.zeros_like(Count_dz))
        
    for t in xrange(num_threads):
        idx = workloads[t]
        thread = mp.Process(target=run_one, \
                args=(tstamps[idx], Trace[idx], \
                all_Ps, \
                all_Count_zhs, all_Count_szs, all_Count_dzs, \
                all_count_hs, all_count_zs, all_assign, \
                alpha_zh, beta_zs, beta_zd, kernel.__class__, \
                residency_priors, \
                Theta_zh, Psi_sz, Psi_dz, \
                pairs_lock, available_to_pair, pairs, \
                conditions, previous_encounters_s, \
                previous_encounters_d, working_lock, started, ended, \
                t, num_iter))
        threads.append(thread)

    for t in xrange(num_threads):
        threads[t].start()
    
    #Wait until everyone has started work
    while sum(started) != len(started):
        time.sleep(5)
    
    print('Everyone is working!')

    #Loop to wait all processes. Not the most elegant sollution
    #But it works.
    sleep = True
    while sum(ended) != len(ended):
        #Only one process left, end it by pairing with self.
        if sum(ended) == len(ended) - 1:
            sleep = False #end busy wait, get it over with
            with pairs_lock:
                for t in xrange(num_threads):
                    available_to_pair[t] = False
                    pairs[t] = t
            for t in xrange(num_threads):
                with conditions[t]:
                    conditions[t].notify()
        if sleep:
            time.sleep(5)

    for t in xrange(num_threads):
        threads[t].join()

    print('Everyone is done!')
    for t in xrange(num_threads):
        idx = workloads[t]
        Trace[:, -1][idx] = all_assign[t]
        
    for z in xrange(Count_zh.shape[0]):
        previous_stamps._clear_one(z)
        previous_stamps._extend(z, tstamps[Trace[:, -1] == z])
    
    P = kernel.get_state()
    P[:] = np.array(all_Ps).mean(axis=0)
    kernel.update_state(P)

    Count_zh[:] = np.array(all_Count_zhs).sum(axis=0)
    Count_sz[:] = np.array(all_Count_szs).sum(axis=0)
    Count_dz[:] = np.array(all_Count_dzs).sum(axis=0)
    count_h[:] = np.array(all_count_hs).sum(axis=0)
    count_z[:] = np.array(all_count_zs).sum(axis=0)

    _aggregate(Count_zh, Count_sz, Count_dz, count_h, count_z, \
        alpha_zh, beta_zs, beta_zd, Theta_zh, Psi_sz, Psi_dz)
    
    Theta_zh[:] = Theta_zh / Theta_zh.sum(axis=0)
    Psi_sz[:] = Psi_sz / Psi_sz.sum(axis=0)
    Psi_dz[:] = Psi_dz / Psi_dz.sum(axis=0)

def generate_workload(nh, num_threads, Trace):
    hyperids = np.arange(nh)
    np.random.shuffle(hyperids)
    workloads = []

    for t in xrange(num_threads):
        workloads.append([])

    for h in xrange(nh):
        workloads[h % num_threads].append(hyperids[h])

    for t in xrange(num_threads):
        workload_bool = np.zeros(Trace.shape[0], dtype='bool')
        for i in xrange(len(workloads[t])):
            workload_bool += Trace[:, 0] == workloads[t][i]

        idx_workload = np.asarray(np.where(workload_bool)[0], dtype='i4')
        workloads[t] = idx_workload 
    return workloads 

def fit(trace_fpath, num_topics, alpha_zh, beta_zs, beta_zd, kernel, \
        residency_priors, num_iter, num_threads, from_=0, to=np.inf):
    '''
    Learns the latent topics from a temporal hypergraph trace. Here we do a
    asynchronous learning of the topics similar to AD-LDA. An even number of
    threads is required.

    Parameters
    ----------
    trace_fpath : str
        The path of the trace. Each line should be a \
                (timestamp, hypernode, source, destination) where the \
                timestamp is a long (seconds or milliseconds from epoch).

    num_topics : int
        The number of latent spaces to learn

    alpha_zh : float
        The value of the alpha_zh hyperparameter

    beta_zs : float
        The value of the beta_zs (beta) hyperaparameter

    beta_zd : float
        The value of the beta_zd (beta') hyperparameter
    
    kernel : Kernel object
        The kernel to use

    residency_priors : array of float
        The kernel hyper parameters

    num_iter : int
        The number of iterations to learn the model from

    num_threads : int
        The number of threads to use, must be even

    Returns
    -------
    
    TODO: explain this better. For the time being, see the keys of the dict.
    A dictionary with the results.
    '''
    tstamps, Trace, previous_stamps, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(trace_fpath, num_topics, num_iter, from_, \
            to)
    
    workloads = generate_workload(Count_zh.shape[1], num_threads, Trace)
    parallel_fit(tstamps, Trace, previous_stamps, \
            Count_zh, Count_sz, Count_dz, count_h, count_z, alpha_zh, beta_zs, \
            beta_zd, kernel, residency_priors, prob_topics_aux, Theta_zh, \
            Psi_sz, Psi_dz, num_iter, workloads, num_threads)
   
    rv = prepare_results(trace_fpath, num_topics, alpha_zh, beta_zs, beta_zd, \
            kernel, residency_priors, num_iter, -1, tstamps, Trace, \
            Count_zh, Count_sz, Count_dz, count_h, \
            count_z, prob_topics_aux, Theta_zh, Psi_sz, Psi_dz, hyper2id, \
            source2id, dest2id, from_, to)

    rv['num_threads'] = np.asarray([num_threads])
    rv['algorithm'] = np.asarray(['parallel gibbs + em'])
    return rv
