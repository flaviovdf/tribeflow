#-*- coding: utf8
from __future__ import division, print_function

from node_sherlock import dataio
from node_sherlock import StampLists

from node_sherlock.learn import prepare_results 

from mpi4py import MPI

from node_sherlock.plearn import dispatch_jobs
from node_sherlock.plearn import fetch_results
from node_sherlock.plearn import generate_workload
from node_sherlock.plearn import manage
from node_sherlock.plearn import Msg

import _learn
import numpy as np

def finalize_splits(nz, n_splits, splitted, tstamps, Trace, nh, ns, nd, kernel):
    
    new_nz = nz + n_splits #len(set(splitted)) 
    new_P = [row for row in kernel.get_state()]
    for _ in xrange(n_splits):
        new_P.append(kernel.get_priors())

    Trace[:, -1] = splitted
    
    #Populate new counts
    Count_zh_new = np.zeros(shape=(new_nz, nh), dtype='i4')
    Count_sz_new = np.zeros(shape=(ns, new_nz), dtype='i4')
    Count_dz_new = np.zeros(shape=(nd, new_nz), dtype='i4')
    count_z_new = np.zeros(new_nz, dtype='i4')
    count_h_new = np.zeros(nh, dtype='i4')
    
    _learn.fast_populate(Trace, Count_zh_new, Count_sz_new, Count_dz_new, \
            count_h_new, count_z_new)
    
    new_stamps = StampLists(new_nz)
    for z in xrange(new_nz):
        idx = Trace[:, -1] == z
        topic_stamps = tstamps[idx]
        new_stamps._extend(z, topic_stamps)

    return Trace, Count_zh_new, Count_sz_new, Count_dz_new, \
            count_z_new, new_stamps, np.array(new_P)

def split(tstamps, Trace, previous_stamps, Count_zh, Count_sz, Count_dz, \
        count_h, count_z, alpha_zh, beta_zs, beta_zd, ll_per_z, kernel, \
        perc=0.05, min_stamps=50):
    
    nz = Count_zh.shape[0]
    nh = Count_zh.shape[1]
    ns = Count_sz.shape[0]
    nd = Count_dz.shape[0]
    
    assert nz == ll_per_z.shape[0]
    idx_int_all = np.arange(Trace.shape[0], dtype='i4')
    
    #Initiate auxiliary matrices
    Count_zh_spl = np.zeros(shape=(nz + 1, nh), dtype='i4')
    Count_sz_spl = np.zeros(shape=(ns, nz + 1), dtype='i4')
    Count_dz_spl = np.zeros(shape=(nd, nz + 1), dtype='i4')
    count_z_spl = np.zeros(nz + 1, dtype='i4')

    Count_zh_spl[:-1, :] = Count_zh
    Count_sz_spl[:, :-1] = Count_sz
    Count_dz_spl[:, :-1] = Count_dz
    count_z_spl[:-1] = count_z

    ll_per_z_new = np.zeros(nz + 1, dtype='f8')
    ll_per_z_new[:-1] = ll_per_z

    new_stamps = StampLists(nz + 1) 
    for z in xrange(nz):
        new_stamps._extend(z, previous_stamps._get_all(z))
    
    splitted = Trace[:, -1].copy()
    shift = 0

    #Do the splits per topic
    for z in xrange(nz):
        #Candidates for removal
        topic_stamps = np.asanyarray(previous_stamps._get_all(z))
        idx = Trace[:, -1] == z
        assert topic_stamps.shape[0] == idx.sum()
        
        argsrt = topic_stamps.argsort()
        top = int(np.ceil(perc * topic_stamps.shape[0]))
        
        #If not at least 50, exit, not enough for a CCDF estimation
        if top < min_stamps:
            continue
        
        #Populate stamps
        new_stamps._clear_one(z)
        new_stamps._clear_one(nz)
        new_stamps._extend(z, topic_stamps[:-top])
        new_stamps._extend(nz, topic_stamps[-top:])
        
        #Split topic on the Trace. The trace has to be sorted by timestamp!!
        old_assign = Trace[:, -1][idx].copy()
        new_assign = Trace[:, -1][idx].copy()
        new_assign[-top:] = nz
        Trace[:, -1][idx] = new_assign
        
        #Update matrices. Can't really vectorize this :(
        for h, s, d, _ in Trace[idx][-top:]:
            Count_zh_spl[z, h] -= 1
            Count_sz_spl[s, z] -= 1
            Count_dz_spl[d, z] -= 1
            count_z_spl[z] -= 1
            
            Count_zh_spl[nz, h] += 1
            Count_sz_spl[s, nz] += 1
            Count_dz_spl[d, nz] += 1
            count_z_spl[nz] += 1

        #New LL
        ll_per_z_new[z] = 0
        ll_per_z_new[-1] = 0
        
        idx_int = idx_int_all[idx]
        _learn.quality_estimate(tstamps, Trace, \
                new_stamps, Count_zh_spl, Count_sz_spl, Count_dz_spl, count_h, \
                count_z_spl, alpha_zh, beta_zs, beta_zd, \
                ll_per_z_new, idx_int, kernel)
        
        if ll_per_z_new.sum() > ll_per_z.sum():
            new_assign[-top:] = nz + shift
            splitted[idx] = new_assign
            shift += 1

        #Revert trace
        new_stamps._clear_one(z)
        new_stamps._clear_one(nz)
        new_stamps._extend(z, previous_stamps._get_all(z))

        Count_zh_spl[:-1, :] = Count_zh
        Count_sz_spl[:, :-1] = Count_sz
        Count_dz_spl[:, :-1] = Count_dz
        count_z_spl[:-1] = count_z

        Count_zh_spl[-1, :] = 0
        Count_sz_spl[:, -1] = 0
        Count_dz_spl[:, -1] = 0
        count_z_spl[-1] = 0
        
        ll_per_z_new[z] = ll_per_z[z]
        ll_per_z_new[-1] = 0
        Trace[:, -1][idx] = old_assign
    
    return finalize_splits(nz, shift, splitted, tstamps, Trace, nh, ns, nd, \
            kernel)

def correlate_counts(Count_zh, Count_sz, Count_dz, count_h, count_z, \
        alpha_zh, beta_zs, beta_zd):
    
    #Create Probabilities
    Theta_zh = np.zeros_like(Count_zh, dtype='f8')
    Psi_sz = np.zeros_like(Count_sz, dtype='f8')
    Psi_dz = np.zeros_like(Count_dz, dtype='f8')
    
    _learn._aggregate(Count_zh, Count_sz, Count_dz, count_h, count_z, \
            alpha_zh, beta_zs, beta_zd, Theta_zh, Psi_sz, Psi_dz)

    Theta_hz = Theta_zh.T * count_z
    Theta_hz = Theta_hz / Theta_hz.sum(axis=0)
    Psi_sz = Psi_sz / Psi_sz.sum(axis=0)
    Psi_dz = Psi_dz / Psi_dz.sum(axis=0)
    
    #Similarity between every probability
    C = np.cov(Theta_hz.T) + np.cov(Psi_sz.T) + np.cov(Psi_dz.T)
    C /= 3
    #Remove lower diag (symmetric)
    C = np.triu(C, 1)
    return C

def finalize_merge(nz, to_merge, tstamps, Trace, nh, ns, nd, kernel):
    
    new_P_dict = dict((i, row) for i, row in enumerate(kernel.get_state()))
    for z1, z2 in to_merge:
        idx = Trace[:, -1] == z2
        del new_P_dict[z2]
        Trace[:, -1][idx] = z1
    
    new_P = []
    for i in sorted(new_P_dict):
        new_P.append(new_P_dict[i])

    #Make sure new trace has contiguous ids
    new_nz = nz - len(to_merge)
    old_ids = sorted(set(Trace[:, -1]))
    new_ids = range(new_nz)
    
    new_assign = Trace[:, -1].copy()
    for i, z in enumerate(old_ids):
        idx = Trace[:, -1] == z
        new_assign[idx] = new_ids[i]
    Trace[:, -1] = new_assign

    #Populate new counts
    Count_zh_new = np.zeros(shape=(new_nz, nh), dtype='i4')
    Count_sz_new = np.zeros(shape=(ns, new_nz), dtype='i4')
    Count_dz_new = np.zeros(shape=(nd, new_nz), dtype='i4')
    count_z_new = np.zeros(new_nz, dtype='i4')
    count_h_new = np.zeros(nh, dtype='i4')

    _learn.fast_populate(Trace, Count_zh_new, Count_sz_new, Count_dz_new, \
            count_h_new, count_z_new)
    
    new_stamps = StampLists(new_nz)
    for z in xrange(new_nz):
        idx = Trace[:, -1] == z
        topic_stamps = tstamps[idx]
        new_stamps._extend(z, topic_stamps)

    return Trace, Count_zh_new, Count_sz_new, Count_dz_new, \
            count_z_new, new_stamps, np.array(new_P)

def merge(tstamps, Trace, previous_stamps, Count_zh, Count_sz, Count_dz, \
        count_h, count_z, alpha_zh, beta_zs, beta_zd, ll_per_z, kernel):

    nz = Count_zh.shape[0]
    nh = Count_zh.shape[1]
    ns = Count_sz.shape[0]
    nd = Count_dz.shape[0]
    
    idx_int_all = np.arange(Trace.shape[0], dtype='i4')

    #Get the nz most similar
    C = correlate_counts(Count_zh, Count_sz, Count_dz, count_h, count_z, \
            alpha_zh, beta_zs, beta_zd)
    
    #k = int(np.ceil(np.sqrt(nz)))
    idx_dim1, idx_dim2 = \
            np.unravel_index(C.flatten().argsort()[-nz:], C.shape)
    top_sims = zip(idx_dim1, idx_dim2)

    #New info
    new_stamps = previous_stamps.copy()
    Count_zh_mrg = Count_zh.copy()
    Count_sz_mrg = Count_sz.copy()
    Count_dz_mrg = Count_dz.copy()
    count_z_mrg = count_z.copy()

    #Test merges
    merged = set()
    accepted = set()
    for z1, z2 in top_sims:
        if z1 in merged or z2 in merged:
            continue
        
        if C[z1, z2] <= 0: #already at nonsimilar
            break
    
        Count_zh_mrg[:] = Count_zh
        Count_sz_mrg[:] = Count_sz
        Count_dz_mrg[:] = Count_dz
        count_z_mrg[:] = count_z
        
        #Merge z1 and z2
        Count_zh_mrg[z1] += Count_zh[z2]
        Count_sz_mrg[:, z1] += Count_sz[:, z2]
        Count_dz_mrg[:, z1] += Count_dz[:, z2]
        count_z_mrg[z1] += count_z[z2]

        #Remove z2
        Count_zh_mrg[z2] = 0
        Count_sz_mrg[:, z2] = 0
        Count_dz_mrg[:, z2] = 0
        count_z_mrg[z2] = 0
        
        idx = Trace[:, -1] == z2
        Trace[:, -1][idx] = z1
        
	#get stamps for llhood
        idx_int = idx_int_all[idx]
        new_stamps._extend(z1, previous_stamps._get_all(z2))
        new_stamps._clear_one(z2)

        #New likelihood
        ll_per_z_new = ll_per_z.copy()
        ll_per_z_new[z2] = 0

        _learn.quality_estimate(tstamps, Trace, \
                new_stamps, Count_zh_mrg, Count_sz_mrg, Count_dz_mrg, count_h, \
                count_z_mrg, alpha_zh, beta_zs, beta_zd, \
                ll_per_z_new, idx_int, kernel)

        if ll_per_z_new.sum() > ll_per_z.sum():
            merged.add(z1)
            merged.add(z2)
            accepted.add((z1, z2))
        
        #Revert trace
        Trace[:, -1][idx] = z2
        new_stamps._clear_one(z1)
        new_stamps._clear_one(z2)
        new_stamps._extend(z1, previous_stamps._get_all(z1))
        new_stamps._extend(z2, previous_stamps._get_all(z2))
    
    return finalize_merge(nz, accepted, tstamps, Trace, nh, ns, nd, kernel) 

def fit(trace_fpath, num_topics, alpha_zh, beta_zs, beta_zd, kernel, \
        residency_priors, num_iter, num_batches, from_=0, to=np.inf):
    '''
    Learns the latent topics from a temporal hypergraph trace. Here we do a
    asynchronous learning of the topics similar to AD-LDA, as well as the 
    dynamic topic expansion/pruing.

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
    
    num_batches : int
        Defines the number of batches of size num_iter 

    Returns
    -------
    
    TODO: explain this better. For the time being, see the keys of the dict.
    A dictionary with the results.
    '''
    assert num_batches >= 2 
    comm = MPI.COMM_WORLD
    num_workers = comm.size - 1

    tstamps, Trace, previous_stamps, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(trace_fpath, num_topics, num_iter, \
            from_, to)
    
    workloads = generate_workload(Count_zh.shape[1], num_workers, Trace)
    all_idx = np.arange(Trace.shape[0], dtype='i4')
    
    for batch in xrange(num_batches):
        print('Now at batch', batch)
        for worker_id in xrange(1, num_workers + 1):
            comm.send(num_iter, dest=worker_id, tag=Msg.LEARN.value)
    
        dispatch_jobs(tstamps, Trace, Count_zh, Count_sz, Count_dz, count_h, \
                count_z, alpha_zh, beta_zs, beta_zd, kernel, residency_priors, \
                workloads, num_workers, comm)
        manage(comm, num_workers)
        fetch_results(comm, num_workers, workloads, tstamps, Trace, \
                previous_stamps, Count_zh, Count_sz, Count_dz, count_h, \
                count_z, alpha_zh, beta_zs, beta_zd, Theta_zh, Psi_sz, \
                Psi_dz, kernel)

        print('Split')
        ll_per_z = np.zeros(count_z.shape[0], dtype='f8')
        _learn.quality_estimate(tstamps, Trace, previous_stamps, \
                Count_zh, Count_sz, Count_dz, count_h, count_z, alpha_zh, \
                beta_zs, beta_zd, ll_per_z, all_idx, kernel)
        Trace, Count_zh, Count_sz, Count_dz, count_z, previous_stamps, \
                P = split(tstamps, Trace, previous_stamps, Count_zh, \
                Count_sz, Count_dz, count_h, count_z, alpha_zh, beta_zs, \
                beta_zd, ll_per_z, kernel)
        kernel = kernel.__class__()
        kernel.build(Trace.shape[0], Count_zh.shape[0], residency_priors)
        if residency_priors.shape[0] > 0:
            kernel.update_state(P)

        print('Merge')
        ll_per_z = np.zeros(count_z.shape[0], dtype='f8')
        _learn.quality_estimate(tstamps, Trace, previous_stamps, \
                Count_zh, Count_sz, Count_dz, count_h, count_z, alpha_zh, \
                beta_zs, beta_zd, ll_per_z, all_idx, kernel)
        Trace, Count_zh, Count_sz, Count_dz, count_z, previous_stamps, \
                P = merge(tstamps, Trace, previous_stamps, Count_zh, \
                Count_sz, Count_dz, count_h, count_z, alpha_zh, beta_zs, \
                beta_zd, ll_per_z, kernel)
        kernel = kernel.__class__()
        kernel.build(Trace.shape[0], Count_zh.shape[0], residency_priors)
        if residency_priors.shape[0] > 0:
            kernel.update_state(P)
 
	Theta_zh = np.zeros(shape=Count_zh.shape, dtype='f8')
	Psi_sz = np.zeros(shape=Count_sz.shape, dtype='f8')
	Psi_dz = np.zeros(shape=Count_dz.shape, dtype='f8')
	if batch == num_batches - 1:
            print('Computing probs')
    	    _learn._aggregate(Count_zh, Count_sz, Count_dz, count_h, count_z, \
                alpha_zh, beta_zs, beta_zd, Theta_zh, Psi_sz, Psi_dz)
    
    for worker_id in xrange(1, num_workers + 1):
        comm.send(num_iter, dest=worker_id, tag=Msg.STOP.value)
    
    rv = prepare_results(trace_fpath, num_topics, alpha_zh, beta_zs, beta_zd, \
            kernel, residency_priors, num_iter, -1, tstamps, Trace, \
            Count_zh, Count_sz, Count_dz, count_h, \
            count_z, prob_topics_aux, Theta_zh, Psi_sz, Psi_dz, hyper2id, \
            source2id, dest2id, from_, to)

    rv['num_workers'] = np.asarray([num_workers])
    rv['num_batches'] = np.asarray([num_batches])
    rv['algorithm'] = np.asarray(['parallel dynamic'])
    return rv
