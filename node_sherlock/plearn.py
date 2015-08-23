#-*- coding: utf8
'''
This module contains the distributed learning approach we employ
on node_sherlock. Through the use of MPI, parallelization can be performed
on multiple machines.
'''
from __future__ import division, print_function

from _learn import _aggregate
from _learn import em
from _learn import fast_populate

from collections import OrderedDict
from enum import Enum

from learn import prepare_results

from node_sherlock import dataio
from node_sherlock import kernels
from node_sherlock.mycollections.stamp_lists import StampLists

from mpi4py import MPI

import cProfile
import numpy as np
import time

Msg = Enum('Msg', ['STARTED', 'FINISHED', 'PAIRME', 'PAIRED', \
        'LEARN', 'SENDRESULTS', 'STOP'])
MASTER = 0
CACHE_SIZE = 1

def paired_update(comm, previous_encounters_s, Count_sz_local, Count_sz_pair, \
        Count_sz_others, P_local, P_pair):
    
    rank = comm.rank
    comm.isend(rank, dest=MASTER, tag=Msg.PAIRME.value)
    pair_id = comm.recv(source=MASTER, tag=Msg.PAIRED.value)
    
    if pair_id == rank: #Paired with self, do nothing
        return False
    
    elif pair_id < rank:
        comm.Recv([Count_sz_pair, MPI.INT], source=pair_id)
        comm.Recv([P_pair, MPI.DOUBLE], source=pair_id)
        
        comm.Send([Count_sz_local, MPI.INT], dest=pair_id)
        comm.Send([P_local, MPI.DOUBLE], dest=pair_id)
    else:
        comm.Send([Count_sz_local, MPI.INT], dest=pair_id)
        comm.Send([P_local, MPI.DOUBLE], dest=pair_id)
        
        comm.Recv([Count_sz_pair, MPI.INT], source=pair_id)
        comm.Recv([P_pair, MPI.DOUBLE], source=pair_id)

    #Update Counts
    #[:] is to avoid copies of arrays. Make sure we dont lose anything
    N_til_s = previous_encounters_s[pair_id]
    Count_sz_others[:] = Count_sz_others + Count_sz_pair - N_til_s

    N_til_s[:] = Count_sz_pair
    P_local[:] = (P_local + P_pair) / 2.0
    
    return True

def receive_workload(comm):
    sizes = np.zeros(5, dtype='i')
    comm.Recv([sizes, MPI.INT], source=MASTER)

    num_lines = sizes[0]
    nz = sizes[1]
    nh = sizes[2]
    ns = sizes[3]
    n_residency_priors = sizes[4]
    
    Count_zh = np.zeros(shape=(nz, nh), dtype='i4') 
    Count_sz = np.zeros(shape=(ns, nz), dtype='i4')
    count_h = np.zeros(shape=(nh, ), dtype='i4')
    count_z = np.zeros(shape=(nz, ), dtype='i4')
    
    tstamps = np.zeros(shape=(num_lines, ), dtype='f8')
    Trace = np.zeros(shape=(num_lines, 4), dtype='i4')

    comm.Recv([tstamps, MPI.DOUBLE], source=MASTER)
    comm.Recv([Trace, MPI.INT], source=MASTER)

    priors = np.zeros(2 + n_residency_priors, dtype='f8')
    comm.Recv([priors, MPI.DOUBLE], source=MASTER)
    
    alpha_zh = priors[0]
    beta_zs = priors[1]
    residency_priors = priors[2:]
    kernel_class = comm.recv(source=MASTER)
    P = np.zeros(shape=(nz, n_residency_priors), dtype='f8')
    comm.Recv([P, MPI.DOUBLE], source=MASTER)

    kernel = kernel_class()
    kernel.build(Trace.shape[0], Count_zh.shape[0], residency_priors)
    if n_residency_priors > 0:
        kernel.update_state(P)
    
    return tstamps, Trace, Count_zh, Count_sz, \
            count_h, count_z, alpha_zh, beta_zs, kernel

def sample(tstamps, Trace, Count_zh, Count_sz_local, \
        count_h, count_z, alpha_zh, beta_zs, kernel, num_iter, comm):
    
    previous_encounters_s = {}
    for other_processor in xrange(1, comm.size):
        previous_encounters_s[other_processor] = np.zeros_like(Count_sz_local)

    stamps = StampLists(Count_zh.shape[0])
    for z in xrange(Count_zh.shape[0]):
        idx = Trace[:, -1] == z
        stamps._extend(z, tstamps[idx])
    
    aux = np.zeros(Count_zh.shape[0], dtype='f8')
 
    Count_sz_pair = np.zeros_like(Count_sz_local)
    Count_sz_others = np.zeros_like(Count_sz_local)
    Count_sz_sum = np.zeros_like(Count_sz_local)

    Theta_zh = np.zeros_like(Count_zh, dtype='f8')
    Psi_sz = np.zeros_like(Count_sz_local, dtype='f8')
    
    can_pair = True
    for i in xrange(num_iter // CACHE_SIZE):
        #Sample from the local counts and encountered counts
        Count_sz_sum[:] = Count_sz_local + Count_sz_others
        count_z[:] = Count_sz_sum.sum(axis=0)
        
        em(tstamps, Trace, stamps, Count_zh, Count_sz_sum, \
                count_h, count_z, alpha_zh, beta_zs, aux, Theta_zh, \
                Psi_sz, CACHE_SIZE, CACHE_SIZE * 2, kernel, False)

        #Update local counts
        Count_sz_local[:] = Count_sz_sum - Count_sz_others
        count_z[:] = Count_sz_local.sum(axis=0)

        #Update expected belief of other processors
        if can_pair:
            P_local = kernel.get_state()
            can_pair = paired_update(comm, previous_encounters_s, \
                    Count_sz_local, Count_sz_pair, Count_sz_others, \
                    P_local, np.zeros_like(P_local))
            kernel.update_state(P_local)

def work():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    #pr = cProfile.Profile()
    #pr.enable()

    while True:
        status = MPI.Status()
        msg = comm.recv(source=MASTER, tag=MPI.ANY_TAG, status=status)
        event = status.Get_tag()

        if event == Msg.LEARN.value:
            comm.isend(rank, dest=MASTER, tag=Msg.STARTED.value)

            num_iter = msg

            tstamps, Trace, Count_zh, Count_sz, count_h, count_z, \
                    alpha_zh, beta_zs, kernel = receive_workload(comm)
            fast_populate(Trace, Count_zh, Count_sz, count_h, \
                    count_z)
            sample(tstamps, Trace, Count_zh, Count_sz, count_h, \
                    count_z, alpha_zh, beta_zs, kernel, num_iter, \
                    comm)
            
            comm.isend(rank, dest=MASTER, tag=Msg.FINISHED.value)
        elif event == Msg.SENDRESULTS.value:
            comm.Send([np.array(Trace[:, -1], order='C'), MPI.INT], dest=MASTER)
            comm.Send([Count_zh, MPI.INT], dest=MASTER)
            comm.Send([Count_sz, MPI.INT], dest=MASTER)
            comm.Send([count_h, MPI.INT], dest=MASTER)
            comm.Send([count_z, MPI.INT], dest=MASTER)
            comm.Send([kernel.get_state(), MPI.DOUBLE], dest=MASTER)
        elif event == Msg.STOP.value:
            break
        else:
            print('Unknown message received', msg, event, Msg(event))

    #pr.disable()
    #pr.dump_stats('worker-%d.pstats' % rank)

def fetch_results(comm, num_workers, workloads, tstamps, Trace, \
        previous_stamps, Count_zh, Count_sz, count_h, count_z, \
        alpha_zh, beta_zs, Theta_zh, Psi_sz, kernel):
    
    Count_zh[:] = 0
    Count_zh_buff = np.zeros_like(Count_zh)

    Count_sz[:] = 0
    Count_sz_buff = np.zeros_like(Count_sz)

    count_h[:] = 0
    count_h_buff = np.zeros_like(count_h)

    count_z[:] = 0
    count_z_buff = np.zeros_like(count_z)

    P = kernel.get_state()
    P[:] = 0
    P_buff = np.zeros_like(P)
        
    for worker_id in xrange(1, num_workers + 1):
        comm.isend(worker_id, dest=worker_id, tag=Msg.SENDRESULTS.value)
        
        idx = workloads[worker_id - 1]
        assign = np.zeros(Trace[idx].shape[0], dtype='i')
        comm.Recv([assign, MPI.INT], source=worker_id)
        Trace[:, -1][idx] = assign

        comm.Recv([Count_zh_buff, MPI.INT], source=worker_id)
        Count_zh += Count_zh_buff
        
        comm.Recv([Count_sz_buff, MPI.INT], source=worker_id)
        Count_sz += Count_sz_buff

        comm.Recv([count_h_buff, MPI.INT], source=worker_id)
        count_h += count_h_buff
        
        comm.Recv([count_z_buff, MPI.INT], source=worker_id)
        count_z += count_z_buff
        
        comm.Recv([P_buff, MPI.DOUBLE], source=worker_id)
        P += P_buff
    
    P[:] = P / num_workers
    kernel.update_state(P)
    Theta_zh[:] = 0
    Psi_sz[:] = 0

    _aggregate(Count_zh, Count_sz, count_h, count_z, \
        alpha_zh, beta_zs, Theta_zh, Psi_sz)
    
    Theta_zh[:] = Theta_zh / Theta_zh.sum(axis=0)
    Psi_sz[:] = Psi_sz / Psi_sz.sum(axis=0)

    for z in xrange(Count_zh.shape[0]):
        previous_stamps._clear_one(z)
        previous_stamps._extend(z, tstamps[Trace[:, -1] == z])

def manage(comm, num_workers):
    available_to_pair = -1
    finished = {}
    num_finished = 0
    
    for worker_id in xrange(1, num_workers + 1):
        finished[worker_id] = False

    while num_finished != num_workers:
        status = MPI.Status()
        worker_id = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, \
                status=status)
        event = status.Get_tag()

        if event == Msg.STARTED.value:
            print('Worker', worker_id, 'is working!')
        
        elif event == Msg.PAIRME.value:
            if num_finished == num_workers - 1: #only 1 working, pair with self
                comm.isend(worker_id, dest=worker_id, tag=Msg.PAIRED.value)
            else:
                assert available_to_pair != worker_id
                if available_to_pair == -1:
                    available_to_pair = worker_id
                else:
                    comm.isend(available_to_pair, dest=worker_id, \
                            tag=Msg.PAIRED.value)
                    comm.isend(worker_id, dest=available_to_pair, \
                            tag=Msg.PAIRED.value)
                    available_to_pair = -1
        elif event == Msg.FINISHED.value:
            print('Worker', worker_id, 'has finished it\'s iterations!')
            finished[worker_id] = True
            num_finished += 1
            
            #wake up last worker if it's waiting for a pair
            if num_finished == num_workers - 1 and available_to_pair != -1:
                comm.isend(available_to_pair, dest=available_to_pair, \
                        tag=Msg.PAIRED.value)
        else:
            print(0, 'Unknown message received', worker_id, event, Msg(event))

def dispatch_jobs(tstamps, Trace, Count_zh, Count_sz, count_h, \
        count_z, alpha_zh, beta_zs, kernel, residency_priors, \
        workloads, num_workers, comm):
    
    for worker_id in xrange(1, num_workers + 1):
        idx = workloads[worker_id - 1]
        
        sizes = np.zeros(5, dtype='i')
        sizes[0] = Trace[idx].shape[0] 
        sizes[1] = Count_zh.shape[0]
        sizes[2] = Count_zh.shape[1]
        sizes[3] = Count_sz.shape[0]
        sizes[4] = residency_priors.shape[0]
        
        comm.Send([sizes, MPI.INT], dest=worker_id)
        comm.Send([tstamps[idx], MPI.INT], dest=worker_id)
        comm.Send([Trace[idx], MPI.INT], dest=worker_id)

        priors = np.zeros(2 + residency_priors.shape[0], dtype='f8')
        priors[0] = alpha_zh
        priors[1] = beta_zs
        priors[2:] = residency_priors

        comm.Send([priors, MPI.DOUBLE], dest=worker_id)
        comm.send(kernel.__class__, dest=worker_id)
        comm.Send([kernel.get_state(), MPI.DOUBLE], dest=worker_id) 

def generate_workload(nh, num_workers, Trace):
    hyperids = np.arange(nh)
    np.random.shuffle(hyperids)

    hyper2worker = {}
    for i in xrange(nh):
        worker = i % num_workers
        hyper2worker[hyperids[i]] = worker
    
    workloads = np.zeros(shape=(num_workers, Trace.shape[0]), dtype='bool')
    for i in xrange(Trace.shape[0]):
        h = Trace[i, 0]
        worker = hyper2worker[h]
        workloads[worker, i] = True
    
    assert workloads.sum() == Trace.shape[0]
    return workloads

def fit(trace_fpath, num_topics, alpha_zh, beta_zs, kernel, residency_priors, \
        num_iter, from_=0, to=np.inf):
    '''
    Learns the latent topics from a temporal hypergraph trace. Here we do a
    asynchronous learning of the topics similar to AD-LDA. An even number of
    threads is required.

    Parameters
    ----------
    trace_fpath : str
        The path of the trace. Each line should be a
                (timestamp, hypernode, source, destination) where the
                timestamp is a long (seconds or milliseconds from epoch).

    num_topics : int
        The number of latent spaces to learn

    alpha_zh : float
        The value of the alpha_zh hyperparameter

    beta_zs : float
        The value of the beta_zs (beta) hyperaparameter

    kernel : Kernel object
        The kernel to use

    residency_priors : array of float
        The kernel hyper parameters

    num_iter : int
        The number of iterations to learn the model from

    Returns
    -------
    
    TODO: explain this better. For the time being, see the keys of the dict.
    A dictionary with the results.
    '''
    comm = MPI.COMM_WORLD
    num_workers = comm.size - 1

    tstamps, Trace, previous_stamps, Count_zh, Count_sz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            hyper2id, source2id = \
            dataio.initialize_trace(trace_fpath, num_topics, num_iter, \
            from_, to)
    for worker_id in xrange(1, num_workers + 1):
        comm.send(num_iter, dest=worker_id, tag=Msg.LEARN.value)
    
    workloads = generate_workload(Count_zh.shape[1], num_workers, Trace)
    dispatch_jobs(tstamps, Trace, Count_zh, Count_sz, count_h, \
            count_z, alpha_zh, beta_zs, kernel, residency_priors, \
            workloads, num_workers, comm)
    manage(comm, num_workers)
    fetch_results(comm, num_workers, workloads, tstamps, Trace, previous_stamps,\
            Count_zh, Count_sz, count_h, count_z, alpha_zh, \
            beta_zs, Theta_zh, Psi_sz, kernel)

    for worker_id in xrange(1, num_workers + 1):
        comm.send(worker_id, dest=worker_id, tag=Msg.STOP.value)

    rv = prepare_results(trace_fpath, num_topics, alpha_zh, beta_zs, \
            kernel, residency_priors, num_iter, -1, tstamps, \
            Trace, Count_zh, Count_sz, count_h, \
            count_z, prob_topics_aux, Theta_zh, Psi_sz, hyper2id, \
            source2id, from_, to)

    rv['num_workers'] = np.asarray([num_workers])
    rv['algorithm'] = np.asarray(['parallel gibbs + em'])
    return rv
