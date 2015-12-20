#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False

from __future__ import print_function, division

from tribeflow.mycollections.stamp_lists cimport StampLists

from tribeflow.kernels.base cimport Kernel

from tribeflow.sorting.binsearch cimport bsp
from tribeflow.sorting.introsort cimport sort

import numpy as np

cdef class ECCDFKernel(Kernel):

    cdef double[:, ::1] P
    cdef double[::1] priors
    cdef int sort_on_mstep

    def __init__(self, sort_on_mstep=False):
        self.sort_on_mstep = int(sort_on_mstep)

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
    
    def _pdf(self, double x, int z, StampLists stamps):
        return self.pdf(x, z, stamps)

    cdef void mstep(self, StampLists stamps) nogil:
        if self.sort_on_mstep == 0:
            return
        
        cdef int z = 0
        cdef int n = 0
        for z in xrange(self.P.shape[0]):
            n = stamps.size(z)
            sort(stamps.get_all(z), n)
    
    def _mstep(self, StampLists stamps):
        self.mstep(stamps)

    def get_priors(self):
        return np.array(self.priors)
 
    def get_state(self):
        return np.array(self.P)

    def update_state(self, double[:, ::1] P):
        assert P.shape[0] == self.P.shape[0]
        assert P.shape[1] == self.P.shape[1]
        self.P = P
