#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False

from __future__ import division, print_function

from node_sherlock._stamp_lists cimport StampLists
from node_sherlock.myrandom.random cimport rand
from node_sherlock.kernels.base cimport Kernel

import numpy as np

cdef extern from 'math.h':
    double log(double) nogil

cdef extern from 'stdio.h':
    int printf(char *, ...) nogil

cdef int bsp(double *array, double value, int n) nogil:
    '''
    Finds the first element in the array where the given is OR should have been
    in the given array. This is simply a binary search, but if the element is
    not found we return the index where it should have been at.
    '''

    cdef int lower = 0
    cdef int upper = n - 1 #closed interval
    cdef int half = 0
    cdef int idx = -1 
 
    while upper >= lower:
        half = lower + ((upper - lower) // 2)
        if value == array[half]:
            idx = half
            break
        elif value > array[half]:
            lower = half + 1
        else:
            upper = half - 1
    
    if idx == -1: #Element not found, return where it should be
        idx = lower

    return idx

cdef void average(double[:,::1] Theta_zh, double[:,::1] Psi_sz, \
        double[:,::1] Psi_dz, int n) nogil:

    cdef int nz = Theta_zh.shape[0]
    cdef int nh = Theta_zh.shape[1]
    cdef int ns = Psi_sz.shape[0]
    cdef int nd = Psi_dz.shape[0]
    
    cdef int z = 0
    cdef int h = 0
    cdef int s = 0 
    cdef int d = 0
    for z in xrange(nz):
        for h in xrange(nh):
            Theta_zh[z, h] /= n

        for s in xrange(ns):
            Psi_sz[s, z] /= n
    
        for d in xrange(nd):
            Psi_dz[d, z] /= n

def _average(Theta_zh, Psi_sz, Psi_dz, n):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    average(Theta_zh, Psi_sz, Psi_dz, n)

cdef void aggregate(int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[:,::1] Count_dz, int[::1] count_h, int[::1] count_z, \
        double alpha_zh, double beta_zs, double beta_zd, \
        double[:,::1] Theta_zh, double[:,::1] Psi_sz, \
        double[:,::1] Psi_dz) nogil:
    
    cdef int nz = Theta_zh.shape[0]
    cdef int nh = Theta_zh.shape[1]
    cdef int ns = Psi_sz.shape[0]
    cdef int nd = Psi_dz.shape[0]
    
    cdef int z = 0
    cdef int h = 0
    cdef int s = 0 
    cdef int d = 0
    for z in xrange(nz):
        for h in xrange(nh):
            Theta_zh[z, h] += dir_posterior(Count_zh[z, h], \
                    count_h[h], nz, alpha_zh) 

        for s in xrange(ns):
            Psi_sz[s, z] += dir_posterior(Count_sz[s, z], \
                    count_z[z], ns, beta_zs) 
    
        for d in xrange(nd):
            Psi_dz[d, z] += dir_posterior(Count_dz[d, z], \
                    count_z[z], nd, beta_zd)

def _aggregate(Count_zh, Count_sz, Count_dz, count_h, count_z, \
        alpha_zh, beta_zs, beta_zd, Theta_zh, Psi_sz, Psi_dz, ):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    aggregate(Count_zh, Count_sz, Count_dz, count_h, count_z, \
        alpha_zh, beta_zs, beta_zd, Theta_zh, Psi_sz, Psi_dz)

cdef double dir_posterior(double joint_count, double global_count, \
        double num_occurences, double smooth) nogil:

    cdef double numerator = smooth + joint_count
    cdef double denominator = global_count + (smooth * num_occurences)
    
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def _dir_posterior(joint_count, global_count, num_occurences, smooth):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    return dir_posterior(joint_count, global_count, num_occurences, smooth)

cdef int sample(double tstamp, int hyper, int source, int dest, \
        StampLists previous_stamps, \
        int[:,::1] Count_zh, int[:,::1] Count_sz, int[:,::1] Count_dz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double beta_zd, double[::1] prob_topics_aux, Kernel kernel) nogil:
    
    cdef int nz = prob_topics_aux.shape[0]
    cdef int ns = Count_sz.shape[0]
    cdef int nd = Count_dz.shape[0]
    cdef int z = 0
    cdef int nstamps = 0
    
    for z in xrange(nz):
        nstamps = previous_stamps.size(z)
        prob_topics_aux[z] = kernel.pdf(tstamp, z, previous_stamps)
        prob_topics_aux[z] = prob_topics_aux[z] * \
            dir_posterior(Count_zh[z, hyper], count_h[hyper], nz, alpha_zh) * \
            dir_posterior(Count_sz[source, z], count_z[z], ns, beta_zs) * \
            dir_posterior(Count_dz[dest, z], count_z[z], nd, beta_zd)

        #accumulate multinomial parameters
        if z >= 1:
            prob_topics_aux[z] += prob_topics_aux[z - 1]

    cdef double u = rand() * prob_topics_aux[nz - 1]
    cdef int new_topic = bsp(&prob_topics_aux[0], u, nz)
    return new_topic

def _sample(tstamp_idx, hyper, source, dest, previous_stamps, \
        Count_zh, Count_sz, Count_dz, count_h, count_z, alpha_zh, beta_zs, \
        beta_zd, prob_topics_aux, kernel):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    return sample(tstamp_idx, hyper, source, dest, previous_stamps, \
            Count_zh, Count_sz, Count_dz, count_h, count_z, \
            alpha_zh, beta_zs, beta_zd, prob_topics_aux, kernel)

cdef void e_step(double[::1] tstamps, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, \
        int[:,::1] Count_sz, int[:,::1] Count_dz, int[::1] count_h, \
        int[::1] count_z, double alpha_zh, double beta_zs, \
        double beta_zd, double[::1] prob_topics_aux, Kernel kernel) nogil:
    
    cdef double tstamp     
    cdef int hyper, source, dest, old_topic
    cdef int new_topic
    cdef int i

    for i in xrange(Trace.shape[0]):
        tstamp = tstamps[i]
        hyper = Trace[i, 0]
        source = Trace[i, 1]
        dest = Trace[i, 2]
        old_topic = Trace[i, 3]

        Count_zh[old_topic, hyper] -= 1
        Count_sz[source, old_topic] -= 1
        Count_dz[dest, old_topic] -= 1
        count_h[hyper] -= 1
        count_z[old_topic] -= 1

        new_topic = sample(tstamp, hyper, source, dest, \
                previous_stamps, \
                Count_zh, Count_sz, Count_dz, count_h, count_z, \
                alpha_zh, beta_zs, beta_zd, \
                prob_topics_aux, kernel)
        
        Trace[i, 3] = new_topic
        Count_zh[new_topic, hyper] += 1
        Count_sz[source, new_topic] += 1
        Count_dz[dest, new_topic] += 1
        count_h[hyper] += 1
        count_z[new_topic] += 1

def _e_step(tstamps, Trace, previous_stamps, Count_zh, Count_sz, \
        Count_dz, count_h, count_z, alpha_zh, beta_zs, beta_zd, \
        prob_topics_aux, kernel):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    e_step(tstamps, Trace, previous_stamps, Count_zh, Count_sz, \
            Count_dz, count_h, count_z, alpha_zh, beta_zs, beta_zd, \
            prob_topics_aux, kernel)

cdef void m_step(double[::1] tstamps, int[:,::1] Trace, \
        StampLists previous_stamps, Kernel kernel) nogil:
    
    previous_stamps.clear()
    cdef int topic
    cdef double tstamp
    cdef int i
    for i in xrange(Trace.shape[0]):
        tstamp = tstamps[i]
        topic = Trace[i, 3]
        previous_stamps.append(topic, tstamp)
    kernel.mstep(previous_stamps)

cdef void col_normalize(double[:,::1] X) nogil:
    
    cdef double sum_ = 0
    cdef int i, j
    for j in xrange(X.shape[1]):
        sum_ = 0

        for i in xrange(X.shape[0]):
            sum_ += X[i, j]

        for i in xrange(X.shape[0]):
            if sum_ > 0:
                X[i, j] = X[i, j] / sum_
            else:
                X[i, j] = 1.0 / X.shape[0]

cdef void fast_em(double[::1] tstamps, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[:,::1] Count_dz, int[::1] count_h, int[::1] count_z, double alpha_zh, \
        double beta_zs, double beta_zd, \
        double[::1] prob_topics_aux, double[:,::1] Theta_zh, \
        double[:,::1] Psi_sz, double[:,::1] Psi_dz, int num_iter, \
        int burn_in, Kernel kernel) nogil:

    cdef int useful_iters = 0
    cdef int i
    for i in xrange(num_iter):
        e_step(tstamps, Trace, previous_stamps, \
                Count_zh, Count_sz, Count_dz, count_h, count_z, \
                alpha_zh, beta_zs, beta_zd, \
                prob_topics_aux, kernel)
        m_step(tstamps, Trace, previous_stamps, kernel)
        
        #average everything out after burn_in
        if i >= burn_in:
            aggregate(Count_zh, Count_sz, Count_dz, \
                    count_h, count_z, alpha_zh, beta_zs, beta_zd, \
                    Theta_zh, Psi_sz, Psi_dz)
            useful_iters += 1

    average(Theta_zh, Psi_sz, Psi_dz, useful_iters)
    col_normalize(Theta_zh)
    col_normalize(Psi_sz)
    col_normalize(Psi_dz)

def em(tstamps, Trace, previous_stamps, Count_zh, Count_sz, Count_dz, \
        count_h, count_z, alpha_zh, beta_zs, beta_zd, \
        prob_topics_aux, \
        Theta_zh, Psi_sz, Psi_dz, num_iter, burn_in, kernel):
    
    fast_em(tstamps, Trace, previous_stamps, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, alpha_zh, beta_zs, beta_zd, \
            prob_topics_aux, \
            Theta_zh, Psi_sz, Psi_dz, num_iter, burn_in, kernel)

def fast_populate(int[:,::1] Trace, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[:,::1] Count_dz, int[::1] count_h, int[::1] count_z):
    
    cdef int i, h, s, d, z
    for i in xrange(Trace.shape[0]):
        h = Trace[i, 0]
        s = Trace[i, 1]
        d = Trace[i, 2]
        z = Trace[i, 3]

        Count_zh[z, h] += 1
        Count_sz[s, z] += 1
        Count_dz[d, z] += 1
        count_h[h] += 1
        count_z[z] += 1

def quality_estimate(double[::1] tstamps, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[:,::1] Count_dz, int[::1] count_h, int[::1] count_z, \
        double alpha_zh, double beta_zs, double beta_zd, \
        double[::1] ll_per_z, int[::1] idx, Kernel kernel):

    cdef int nz = Count_zh.shape[0]
    cdef int ns = Count_sz.shape[0]
    cdef int nd = Count_dz.shape[0]
    
    cdef int i = 0
    cdef int h, z, s, d = 0
    cdef double tstamp = 0

    for i in range(idx.shape[0]):
        tstamp = tstamps[idx[i]]
        h = Trace[idx[i], 0]
        s = Trace[idx[i], 1]
        d = Trace[idx[i], 2]
        z = Trace[idx[i], 3]

        ll_per_z[z] += log(kernel.pdf(tstamp, z, previous_stamps)) + \
            log(dir_posterior(Count_zh[z, h], count_h[h], nz, alpha_zh)) + \
            log(dir_posterior(Count_sz[s, z], count_z[z], ns, beta_zs)) + \
            log(dir_posterior(Count_dz[d, z], count_z[z], nd, beta_zd))

def mean_reciprocal_rank(double[::1] tstamps, int[:, ::1] HSDs, \
        StampLists previous_stamps, double[:, ::1] Theta_zh, \
        double[:, ::1] Psi_sz, double[:, ::1] Psi_dz, int[::1] count_z, \
        Kernel kernel):
        
    cdef double dt = 0
    cdef int h = 0
    cdef int s = 0
    cdef int real_d = 0
    cdef int candidate_d = 0

    cdef int z = 0

    cdef double[::1] aux_base = np.zeros(Psi_dz.shape[0], dtype='d')
    cdef double[::1] aux_delta = np.zeros(Psi_dz.shape[0], dtype='d')
    cdef double[::1] aux_full = np.zeros(Psi_dz.shape[0], dtype='d')
    
    cdef double gt_base = 0
    cdef double gt_delta = 0
    cdef double gt_full = 0

    cdef double mrr_base = 0
    cdef double mrr_delta = 0
    cdef double mrr_full = 0
    
    cdef int i = 0
    for i in xrange(HSDs.shape[0]):
        dt = tstamps[i]
        h = HSDs[i, 0]
        s = HSDs[i, 1]
        real_d = HSDs[i, 2]
        
        for candidate_d in xrange(Psi_dz.shape[0]):
            aux_base[candidate_d] = 0.0
            aux_delta[candidate_d] = 0.0
            aux_full[candidate_d] = 0.0

        for z in xrange(Psi_dz.shape[1]):
            for candidate_d in xrange(Psi_dz.shape[0]):
                aux_base[candidate_d] += count_z[z] * Psi_sz[s, z] * \
                        Psi_dz[candidate_d, z] 
                aux_delta[candidate_d] += count_z[z] * Psi_sz[s, z] * \
                        Psi_dz[candidate_d, z] * \
                        kernel.pdf(dt, z, previous_stamps)
                aux_full[candidate_d] += Psi_sz[s, z] * \
                        Psi_dz[candidate_d, z] * Theta_zh[z, h] * \
                        kernel.pdf(dt, z, previous_stamps)
        
        gt_base = 0.0
        gt_delta = 0.0
        gt_full = 0.0

        for candidate_d in xrange(Psi_dz.shape[0]):
            if aux_base[candidate_d] >= aux_base[real_d]:
                gt_base += 1

            if aux_delta[candidate_d] >= aux_delta[real_d]:
                gt_delta += 1

            if aux_full[candidate_d] >= aux_full[real_d]:
                gt_full += 1

        mrr_base += 1.0 / gt_base
        mrr_delta += 1.0 / gt_delta
        mrr_full += 1.0 / gt_full

    mrr_base = mrr_base / HSDs.shape[0]
    mrr_delta = mrr_delta / HSDs.shape[0]
    mrr_full = mrr_full / HSDs.shape[0]
    return mrr_base, mrr_delta, mrr_full
