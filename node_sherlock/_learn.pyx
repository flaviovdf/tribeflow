#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False

from __future__ import division, print_function

from cython.parallel cimport prange

from node_sherlock.mycollections.stamp_lists cimport StampLists
from node_sherlock.myrandom.random cimport rand
from node_sherlock.sorting.binsearch cimport bsp

from node_sherlock.kernels.base cimport Kernel

import numpy as np

cdef extern from 'math.h':
    double log(double) nogil

cdef extern from 'stdio.h':
    int printf(char *, ...) nogil

cdef void average(double[:,::1] Theta_zh, double[:,::1] Psi_sz, int n) nogil:

    cdef int nz = Theta_zh.shape[0]
    cdef int nh = Theta_zh.shape[1]
    cdef int ns = Psi_sz.shape[0]
    
    cdef int z = 0
    cdef int h = 0
    cdef int s = 0 
    for z in xrange(nz):
        for h in xrange(nh):
            Theta_zh[z, h] /= n

        for s in xrange(ns):
            Psi_sz[s, z] /= n

def _average(Theta_zh, Psi_sz, n):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    average(Theta_zh, Psi_sz, n)

cdef void aggregate(int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double[:,::1] Theta_zh, double[:,::1] Psi_sz) nogil:
    
    cdef int nz = Theta_zh.shape[0]
    cdef int nh = Theta_zh.shape[1]
    cdef int ns = Psi_sz.shape[0]
    
    cdef int z = 0
    cdef int h = 0
    cdef int s = 0 
    for z in xrange(nz):
        for h in xrange(nh):
            Theta_zh[z, h] += dir_posterior(Count_zh[z, h], count_h[h], nz, \
                    alpha_zh) 

        for s in xrange(ns):
            Psi_sz[s, z] += dir_posterior(Count_sz[s, z], count_z[z], ns, \
                    beta_zs) 
    
def _aggregate(Count_zh, Count_sz, count_h, count_z, \
        alpha_zh, beta_zs, Theta_zh, Psi_sz):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    aggregate(Count_zh, Count_sz, count_h, count_z, alpha_zh, beta_zs, \
            Theta_zh, Psi_sz)

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

cdef int sample(double dt, int hyper, int source, int dest, \
        StampLists previous_stamps, \
        int[:,::1] Count_zh, int[:,::1] Count_sz, int[::1] count_h, \
        int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] prob_topics_aux, Kernel kernel) nogil:
    
    cdef int nz = Count_zh.shape[0]
    cdef int ns = Count_sz.shape[0]
    cdef int z = 0
    
    for z in xrange(nz):
        prob_topics_aux[z] = kernel.pdf(dt, z, previous_stamps) * \
            dir_posterior(Count_zh[z, hyper], count_h[hyper], nz, alpha_zh) * \
            dir_posterior(Count_sz[source, z], count_z[z], ns, beta_zs) * \
            dir_posterior(Count_sz[dest, z], count_z[z], ns, beta_zs)
            
        #accumulate multinomial parameters
        if z >= 1:
            prob_topics_aux[z] += prob_topics_aux[z - 1]
    
    cdef double u = rand() * prob_topics_aux[nz - 1]
    cdef int new_topic = bsp(&prob_topics_aux[0], u, nz)
    return new_topic

def _sample(dt, hyper, source, dest, previous_stamps, Count_zh, Count_sz, \
        count_h, count_z, alpha_zh, beta_zs, prob_topics_aux, kernel):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    return sample(dt, hyper, source, dest, previous_stamps, Count_zh, \
            Count_sz, count_h, count_z, alpha_zh, beta_zs, prob_topics_aux, \
            kernel)

cdef void e_step(double[::1] dts, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] prob_topics_aux, Kernel kernel) nogil:
    
    cdef double dt    
    cdef int hyper, source, dest, old_topic
    cdef int new_topic
    cdef int i, j
    
    for i in xrange(Trace.shape[0]):
        dt = dts[i]
        hyper = Trace[i, 0]
        source = Trace[i, 1]
        dest = Trace[i, 2]
        old_topic = Trace[i, 3]

        Count_zh[old_topic, hyper] -= 1
        Count_sz[source, old_topic] -= 1
        Count_sz[dest, old_topic] -= 1
        count_h[hyper] -= 1
        count_z[old_topic] -= 1

        new_topic = sample(dt, hyper, source, dest, previous_stamps, Count_zh, \
                Count_sz, count_h, count_z, alpha_zh, beta_zs, \
                prob_topics_aux, kernel)
        Trace[i, 3] = new_topic
        
        Count_zh[new_topic, hyper] += 1
        Count_sz[source, new_topic] += 1
        Count_sz[dest, new_topic] += 1
        count_h[hyper] += 1
        count_z[new_topic] += 1

def _e_step(dts, Trace, previous_stamps, Count_zh, Count_sz, count_h, \
        count_z, alpha_zh, beta_zs, prob_topics_aux, kernel):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    e_step(dts, Trace, previous_stamps, Count_zh, Count_sz, count_h, \
            count_z, alpha_zh, beta_zs, prob_topics_aux, kernel)

cdef void m_step(double[::1] dts, int[:,::1] Trace, \
        StampLists previous_stamps, Kernel kernel) nogil:
    
    previous_stamps.clear()
    cdef int topic
    cdef double dt
    cdef int i
    for i in xrange(Trace.shape[0]):
        dt = dts[i]
        topic = Trace[i, 3]
        previous_stamps.append(topic, dt)
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

cdef void fast_em(double[::1] dts, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] prob_topics_aux, double[:,::1] Theta_zh, \
        double[:,::1] Psi_sz, int num_iter, int burn_in, Kernel kernel) nogil:

    cdef int useful_iters = 0
    cdef int i
    for i in xrange(num_iter):
        e_step(dts, Trace, previous_stamps, Count_zh, Count_sz, count_h, \
                count_z, alpha_zh, beta_zs, prob_topics_aux, kernel)
        m_step(dts, Trace, previous_stamps, kernel)
        
        #average everything out after burn_in
        if i >= burn_in:
            aggregate(Count_zh, Count_sz, \
                    count_h, count_z, alpha_zh, beta_zs, \
                    Theta_zh, Psi_sz)
            useful_iters += 1

    average(Theta_zh, Psi_sz, useful_iters)
    col_normalize(Theta_zh)
    col_normalize(Psi_sz)

def em(dts, Trace, previous_stamps, Count_zh, Count_sz, count_h, count_z, \
        alpha_zh, beta_zs, prob_topics_aux, Theta_zh, Psi_sz, num_iter, \
        burn_in, kernel):
    
    fast_em(dts, Trace, previous_stamps, Count_zh, Count_sz, \
            count_h, count_z, alpha_zh, beta_zs, prob_topics_aux, \
            Theta_zh, Psi_sz, num_iter, burn_in, kernel)

def fast_populate(int[:,::1] Trace, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z):
    
    cdef int i, h, s, d, z
    for i in xrange(Trace.shape[0]):
        h = Trace[i, 0]
        s = Trace[i, 1]
        d = Trace[i, 2]
        z = Trace[i, 3]

        Count_zh[z, h] += 1
        Count_sz[s, z] += 1
        Count_sz[d, z] += 1
        count_h[h] += 1
        count_z[z] += 2

def quality_estimate(double[::1] dts, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] ll_per_z, int[::1] idx, Kernel kernel):

    cdef int nz = Count_zh.shape[0]
    cdef int ns = Count_sz.shape[0]
    
    cdef int i = 0
    cdef int h, z, s, d = 0
    cdef double dt = 0

    for i in xrange(idx.shape[0]):
        dt = dts[idx[i]] 
        h = Trace[idx[i], 0]
        s = Trace[idx[i], 1]
        d = Trace[idx[i], 2]
        z = Trace[idx[i], 3]

        ll_per_z[z] += \
                log(dir_posterior(Count_zh[z, h], count_h[h], nz, alpha_zh)) + \
                log(dir_posterior(Count_sz[s, z], count_z[z], ns, beta_zs)) + \
                log(dir_posterior(Count_sz[d, z], count_z[z], ns, beta_zs)) + \
                log(kernel.pdf(dt, z, previous_stamps))

def reciprocal_rank(double[::1] tstamps, int[:, ::1] HOs, \
        StampLists previous_stamps, double[:, ::1] Theta_zh, \
        double[:, ::1] Psi_sz, int[::1] count_z, Kernel kernel):
        
    cdef double dt = 0
    cdef int h = 0
    cdef int s = 0
    cdef int real_o = 0
    cdef int candidate_o = 0

    cdef int z = 0

    cdef double[::1] aux_base = np.zeros(Psi_sz.shape[0], dtype='d')
    cdef double[::1] aux_delta = np.zeros(Psi_sz.shape[0], dtype='d')
    cdef double[::1] aux_full = np.zeros(Psi_sz.shape[0], dtype='d')
    
    cdef double[:, ::1] rrs = np.zeros(shape=(HOs.shape[0], 3), dtype='d')
    cdef int i = 0
    for i in xrange(HOs.shape[0]):
        dt = tstamps[i + 1] - tstamps[i]
        h = HOs[i, 0]
        real_o = HOs[i, 1]
        
        for candidate_o in prange(Psi_sz.shape[0], schedule='static', nogil=True):
            aux_base[candidate_o] = 0.0
            aux_delta[candidate_o] = 0.0
            aux_full[candidate_o] = 0.0

        for z in xrange(Psi_sz.shape[1]):
            for candidate_o in prange(Psi_sz.shape[0], schedule='static', nogil=True):
                aux_base[candidate_o] += count_z[z] * Psi_sz[s, z] * \
                        Psi_sz[candidate_o, z] 
                aux_delta[candidate_o] += count_z[z] * Psi_sz[s, z] * \
                        Psi_sz[candidate_o, z] * \
                        kernel.pdf(dt, z, previous_stamps)
                aux_full[candidate_o] += Psi_sz[s, z] * \
                        Psi_sz[candidate_o, z] * Theta_zh[z, h] * \
                        kernel.pdf(dt, z, previous_stamps)
        
        for candidate_o in prange(Psi_sz.shape[0], schedule='static', nogil=True):
            if aux_base[candidate_o] >= aux_base[real_o]:
                rrs[i, 0] += 1

            if aux_delta[candidate_o] >= aux_delta[real_o]:
                rrs[i, 1] += 1

            if aux_full[candidate_o] >= aux_full[real_o]:
                rrs[i, 2] += 1
        
        rrs[i, 0] = 1.0 / rrs[i, 0]
        rrs[i, 1] = 1.0 / rrs[i, 1]
        rrs[i, 2] = 1.0 / rrs[i, 2]
    
    return np.array(rrs)
