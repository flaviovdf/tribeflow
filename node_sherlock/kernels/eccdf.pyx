#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False

from __future__ import print_function, division

from node_sherlock._stamp_lists cimport StampLists
from node_sherlock.kernels.base cimport Kernel

import numpy as np

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

cdef class ECCDFKernel(Kernel):

    cdef double[:, ::1] P
    cdef double[::1] priors

    def build(self, int trace_size, int nz, double[::1] priors):
        
        assert priors.shape[0] == 2
        
        self.P = np.zeros(shape=(nz, 2), dtype='d')
        self.priors = priors

        cdef int z
        for z in xrange(nz):
            self.P[z, 0] = priors[0]
            self.P[z, 1] = priors[1]

    cdef double pdf(self, double x, int z, StampLists stamps) nogil:
        cdef double a = self.P[z, 0]
        cdef double b = self.P[z, 1]
        cdef int n = stamps.size(z)

        cdef int loc = bsp(stamps.get_all(z), x, n)
        return (a + n - loc) / (a + b + n)
    
    cdef void mstep(self, StampLists stamps) nogil:
        cdef int nz = self.P.shape[0]
        cdef double total = 0
        cdef int z
        for z in xrange(nz):
            self.P[z, 0] = stamps.size(z)
            total += self.P[z, 0]

        for z in xrange(nz):
            self.P[z, 1] = total

    def get_priors(self):
        return np.array(self.priors)
 
    def get_state(self):
        return np.array(self.P)

    def update_state(self, double[:, ::1] P):
        assert P.shape[0] == self.P.shape[0]
        assert P.shape[1] == self.P.shape[1]
        self.P = P
