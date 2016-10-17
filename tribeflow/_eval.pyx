#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False
from __future__ import division, print_function

from cpython cimport bool
from cython.parallel cimport prange

from tribeflow cimport _learn
from tribeflow.kernels.base cimport Kernel
from tribeflow.mycollections.stamp_lists cimport StampLists

import numpy as np

cdef extern from 'math.h':
    double log(double) nogil

def quality_estimate(double[:,::1] Dts, int[:,::1] Trace, \
        StampLists previous_stamps, int[:,::1] Count_zh, int[:,::1] Count_sz, \
        int[::1] count_h, int[::1] count_z, double alpha_zh, double beta_zs, \
        double[::1] ll_per_z, int[::1] idx, Kernel kernel):

    cdef int nz = Count_zh.shape[0]
    cdef int ns = Count_sz.shape[0]
    
    cdef int i = 0
    cdef int j = 0
    cdef int h = 0
    cdef int z = 0
    cdef int o = 0
    cdef int p = 0
    cdef double dt = 0

    for i in xrange(idx.shape[0]):
        dt = Dts[idx[i], Dts.shape[1] - 1] 
        z = Trace[idx[i], Trace.shape[1] - 1]
        h = Trace[idx[i], 0]
        o = Trace[idx[i], 1]
        ll_per_z[z] += \
            log(_learn.dir_posterior(Count_sz[o, z], count_z[z], ns, beta_zs))

        for j in xrange(2, Trace.shape[1] - 1):
            o = Trace[i, j]
            p = Trace[i, j - 1]
            ll_per_z[z] += \
                log(_learn.dir_posterior(Count_sz[o, z], count_z[z], ns, beta_zs) / \
                (1 - _learn.dir_posterior(Count_sz[p, z], count_z[z], ns, beta_zs)))

        ll_per_z[z] += \
                log(_learn.dir_posterior(Count_zh[z, h], count_h[h], nz, alpha_zh)) + \
                log(kernel.pdf(dt, z, previous_stamps))

def reciprocal_rank(double[:, ::1] Dts, int[:, ::1] HOs, \
        StampLists previous_stamps, double[:, ::1] Theta_zh, \
        double[:, ::1] Psi_sz, int[::1] count_z, Kernel kernel, 
        bool return_probs=False):
    '''
    Computes the reciprocal rank of predictions. Parameter descriptions
    below consider a burst of size `B`.

    Parameters
    ----------
    Dts: with inter event times. Shape is (n_events, B)
    HOs: hyper node (users) with burst (objetcs). Shape is (n_events, B+1). The
    last element in each row is the true object.
    previous_stamps: time_stamp to compute inter event times
    Theta_zh, Psi_sz, count_z: outputs of tribelow
    kernel: inter event kernel
    return_probs: boolean indicating if we should return full probabilities
     (unormalized for each row).
    
    Returns
    -------

    An array with reciprocal ranks and another with probabilities.
    '''
        
    cdef double dt = 0
    cdef int h = 0
    cdef int s = 0
    cdef int real_o = 0
    cdef int candidate_o = 0
    cdef int last_o = 0

    cdef int z = 0
    cdef int ns = Psi_sz.shape[0]
    
    cdef int[::1] mem = np.zeros(Dts.shape[1], dtype='i4')
    cdef double[::1] mem_factor = np.zeros(Psi_sz.shape[1], dtype='d')
    cdef double[::1] p = np.zeros(Psi_sz.shape[0], dtype='d')
    
    cdef double[:] rrs = np.zeros(shape=(HOs.shape[0], ), dtype='d')
    cdef double[:, ::1] predictions = \
            np.zeros(shape=(HOs.shape[0], ns), dtype='d')

    cdef int i, j
    for i in xrange(HOs.shape[0]):
        dt = Dts[i, Dts.shape[1] - 1] 
        h = HOs[i, 0]
        for j in xrange(mem.shape[0]):
            mem[j] = HOs[i, 1 + j]
        real_o = HOs[i, HOs.shape[1] - 1]
        last_o = HOs[i, HOs.shape[1] - 2]
        
        for candidate_o in prange(ns, schedule='static', nogil=True):
            p[candidate_o] = 0.0
        
        for z in xrange(Psi_sz.shape[1]):
            mem_factor[z] = 1.0
            for j in xrange(mem.shape[0]):
                if j == 0:
                    mem_factor[z] *= Psi_sz[mem[j], z]
                else:
                    mem_factor[z] *= \
                            Psi_sz[mem[j], z] / (1.0 - Psi_sz[mem[j-1], z])
            mem_factor[z] *= 1.0 / (1 - Psi_sz[mem[mem.shape[0] - 1], z])

        for z in xrange(Psi_sz.shape[1]):
            for candidate_o in prange(ns, schedule='static', nogil=True):
                p[candidate_o] += mem_factor[z] * \
                        Psi_sz[candidate_o, z] * Theta_zh[z, h] * \
                        kernel.pdf(dt, z, previous_stamps)
        
        for candidate_o in prange(ns, schedule='static', nogil=True):
            if p[candidate_o] >= p[real_o] and candidate_o != last_o:
                rrs[i] += 1
            predictions[i, candidate_o] = p[candidate_o]
    
    if return_probs:
        return np.array(rrs), np.array(predictions)
    else:
        return np.array(rrs)
