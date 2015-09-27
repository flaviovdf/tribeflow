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

cdef inline double dir_posterior(double joint_count, double global_count, \
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

cdef inline int sample(int idx, double[:,::1] Dts, int[:,::1] Trace, \
        StampLists previous_stamps, \
        int[:,::1] Count_zh, int[:,::1] Count_sz, int[::1] count_h, \
        int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] prob_topics_aux, Kernel kernel) nogil:
     
    cdef int nz = Count_zh.shape[0]
    cdef int ns = Count_sz.shape[0]
    cdef int z, j
    cdef int hyper = Trace[idx, 0]
    cdef double dt = Dts[idx, Dts.shape[1] - 1]
    cdef int last_obj = Trace[idx, Trace.shape[1] - 2]
    cdef int obj

    for z in xrange(nz):
        prob_topics_aux[z] = kernel.pdf(dt, z, previous_stamps) * \
            dir_posterior(Count_zh[z, hyper], count_h[hyper], nz, alpha_zh)

        for j in xrange(1, Trace.shape[1] - 1):
            obj = Trace[idx, j]
            prob_topics_aux[z] = prob_topics_aux[z] * \
                    dir_posterior(Count_sz[obj, z], count_z[z], ns, beta_zs)

        prob_topics_aux[z] = prob_topics_aux[z] / \
            (1 - dir_posterior(Count_sz[last_obj, z], count_z[z], ns, beta_zs))
        
        #accumulate multinomial parameters
        if z >= 1:
            prob_topics_aux[z] += prob_topics_aux[z - 1]
    
    cdef double u = rand() * prob_topics_aux[nz - 1]
    cdef int new_topic = bsp(&prob_topics_aux[0], u, nz)
    return new_topic

def _sample(idx, Dts, Trace, previous_stamps, Count_zh, Count_sz, \
        count_h, count_z, alpha_zh, beta_zs, prob_topics_aux, kernel):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    return sample(idx, Dts, Trace, previous_stamps, Count_zh, \
            Count_sz, count_h, count_z, alpha_zh, beta_zs, prob_topics_aux, \
            kernel)

cdef inline void e_step(double[:,::1] Dts, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] prob_topics_aux, Kernel kernel) nogil:
    
    cdef double dt    
    cdef int hyper, obj, old_topic, new_topic
    cdef int i, j
    cdef int mem_size = Dts.shape[0]

    for i in xrange(Trace.shape[0]):
        dt = Dts[i, Dts.shape[1] - 1]
        hyper = Trace[i, 0]
        old_topic = Trace[i, Trace.shape[1] - 1]

        Count_zh[old_topic, hyper] -= 1
        count_h[hyper] -= 1
        
        for j in xrange(1, Trace.shape[1] - 1):
            obj = Trace[i, j]
            Count_sz[obj, old_topic] -= 1
            count_z[old_topic] -= 1
        
        new_topic = sample(i, Dts, Trace, previous_stamps, Count_zh, \
                Count_sz, count_h, count_z, alpha_zh, beta_zs, \
                prob_topics_aux, kernel)
        Trace[i, Trace.shape[1] - 1] = new_topic
        
        Count_zh[new_topic, hyper] += 1
        count_h[hyper] += 1
        
        for j in xrange(1, Trace.shape[1] - 1):
            obj = Trace[i, j]
            Count_sz[obj, new_topic] += 1
            count_z[new_topic] += 1

def _e_step(Dts, Trace, previous_stamps, Count_zh, Count_sz, count_h, \
        count_z, alpha_zh, beta_zs, prob_topics_aux, kernel):
    '''Wrapper used mostly for unit tests. Do not call directly otherwise'''
    e_step(Dts, Trace, previous_stamps, Count_zh, Count_sz, count_h, \
            count_z, alpha_zh, beta_zs, prob_topics_aux, kernel)

cdef inline void m_step(double[:,::1] Dts, int[:,::1] Trace, \
        StampLists previous_stamps, Kernel kernel) nogil:
    
    previous_stamps.clear()
    cdef int topic
    cdef double dt
    cdef int i
    for i in xrange(Trace.shape[0]):
        dt = Dts[i, Dts.shape[1] - 1]
        topic = Trace[i, Trace.shape[1] - 1]
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

cdef void fast_em(double[:,::1] Dts, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] prob_topics_aux, double[:,::1] Theta_zh, \
        double[:,::1] Psi_sz, int num_iter, int burn_in, Kernel kernel) nogil:

    cdef int i
    for i in xrange(num_iter):
        e_step(Dts, Trace, previous_stamps, Count_zh, Count_sz, count_h, \
                count_z, alpha_zh, beta_zs, prob_topics_aux, kernel)
        m_step(Dts, Trace, previous_stamps, kernel)
        
        #average everything out after burn_in
        if i >= burn_in:
            aggregate(Count_zh, Count_sz, \
                    count_h, count_z, alpha_zh, beta_zs, \
                    Theta_zh, Psi_sz)

def em(Dts, Trace, previous_stamps, Count_zh, Count_sz, count_h, count_z, \
        alpha_zh, beta_zs, prob_topics_aux, Theta_zh, Psi_sz, num_iter, \
        burn_in, kernel, average_and_normalize=True):
    
    fast_em(Dts, Trace, previous_stamps, Count_zh, Count_sz, \
            count_h, count_z, alpha_zh, beta_zs, prob_topics_aux, \
            Theta_zh, Psi_sz, num_iter, burn_in, kernel)
    if average_and_normalize:
        if (num_iter - burn_in) > 0:
            average(Theta_zh, Psi_sz, num_iter - burn_in)
        col_normalize(Theta_zh)
        col_normalize(Psi_sz)

def fast_populate(int[:,::1] Trace, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z):
    
    cdef int i, j, h, o, z
    for i in xrange(Trace.shape[0]):
        h = Trace[i, 0]
        z = Trace[i, Trace.shape[1] - 1]

        Count_zh[z, h] += 1
        count_h[h] += 1
        for j in xrange(1, Trace.shape[1] - 1):
            o = Trace[i, j]
            Count_sz[o, z] += 1
            count_z[z] += 1

def quality_estimate(double[:,::1] Dts, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] ll_per_z, int[::1] idx, Kernel kernel):

    cdef int nz = Count_zh.shape[0]
    cdef int ns = Count_sz.shape[0]
    
    cdef int i = 0
    cdef int h, z, o = 0
    cdef double dt = 0

    for i in xrange(idx.shape[0]):
        dt = Dts[idx[i], Dts.shape[1] - 1] 
        h = Trace[idx[i], 0]
        for j in xrange(1, Trace.shape[1] - 1):
            o = Trace[i, j]
            for z in range(nz):
                ll_per_z[z] += \
                    log(dir_posterior(Count_sz[o, z], count_z[z], ns, beta_zs))

        for z in range(nz):
            ll_per_z[z] += \
                    log(dir_posterior(Count_zh[z, h], count_h[h], nz, alpha_zh)) + \
                    log(kernel.pdf(dt, z, previous_stamps))

def reciprocal_rank(double[:, ::1] Dts, int[:, ::1] HOs, \
        StampLists previous_stamps, double[:, ::1] Theta_zh, \
        double[:, ::1] Psi_sz, int[::1] count_z, Kernel kernel):
        
    cdef double dt = 0
    cdef int h = 0
    cdef int s = 0
    cdef int real_o = 0
    cdef int candidate_o = 0
    cdef int last_o = 0

    cdef int z = 0
    
    cdef int[::1] mem = np.zeros(Dts.shape[1], dtype='i4')
    cdef double[::1] mem_factor = np.zeros(Psi_sz.shape[1], dtype='d')
    cdef double[::1] aux_base = np.zeros(Psi_sz.shape[0], dtype='d')
    cdef double[::1] aux_delta = np.zeros(Psi_sz.shape[0], dtype='d')
    cdef double[::1] aux_full = np.zeros(Psi_sz.shape[0], dtype='d')
    
    cdef double[:, ::1] rrs = np.zeros(shape=(HOs.shape[0], 3), dtype='d')
    cdef int i, j
    for i in xrange(HOs.shape[0]):
        dt = Dts[i, 0] 
        h = HOs[i, 0]
        for j in xrange(mem.shape[0]):
            mem[j] = HOs[i, 1 + j]
        real_o = HOs[i, HOs.shape[1] - 1]
        last_o = HOs[i, HOs.shape[1] - 2]
        
        for candidate_o in prange(Psi_sz.shape[0], schedule='static', nogil=True):
            aux_base[candidate_o] = 0.0
            aux_delta[candidate_o] = 0.0
            aux_full[candidate_o] = 0.0
        
        for z in xrange(Psi_sz.shape[1]):
            mem_factor[z] = 1.0
            for j in xrange(mem.shape[0]):
                mem_factor[z] *= Psi_sz[mem[j], z]
        mem_factor[z] *= 1.0 / (1 - Psi_sz[mem[mem.shape[0] - 1], z])

        for z in xrange(Psi_sz.shape[1]):
            for candidate_o in prange(Psi_sz.shape[0], schedule='static', nogil=True):
                aux_base[candidate_o] += count_z[z] * mem_factor[z] * \
                        Psi_sz[candidate_o, z] 
                aux_delta[candidate_o] += count_z[z] * mem_factor[z] * \
                        Psi_sz[candidate_o, z] * \
                        kernel.pdf(dt, z, previous_stamps)
                aux_full[candidate_o] += mem_factor[z] * \
                        Psi_sz[candidate_o, z] * Theta_zh[z, h] * \
                        kernel.pdf(dt, z, previous_stamps)
        
        for candidate_o in prange(Psi_sz.shape[0], schedule='static', nogil=True):
            if aux_base[candidate_o] >= aux_base[real_o] \
                    and candidate_o != last_o:
                rrs[i, 0] += 1

            if aux_delta[candidate_o] >= aux_delta[real_o] \
                    and candidate_o != last_o:
                rrs[i, 1] += 1

            if aux_full[candidate_o] >= aux_full[real_o] \
                    and candidate_o != last_o:
                rrs[i, 2] += 1
        
        rrs[i, 0] = 1.0 / rrs[i, 0]
        rrs[i, 1] = 1.0 / rrs[i, 1]
        rrs[i, 2] = 1.0 / rrs[i, 2]
    
    return np.array(rrs)
