#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False

from __future__ import print_function, division

from tribeflow.mycollections.stamp_lists cimport StampLists
from tribeflow.kernels.base cimport Kernel

import numpy as np

cdef class NoopKernel(Kernel):
    
    cdef double[:, ::1] P
    cdef double[::1] priors

    cdef double pdf(self, double x, int z, StampLists stamps) nogil:
        return 1.0

    def build(self, int trace_size, int nz, double[::1] priors):
        self.P = np.zeros(shape=(0, 0), dtype='d')
        self.priors = priors

    def get_priors(self):
        return np.array(self.priors)
    
    def get_state(self):
        return np.array(self.P)

    def update_state(self, double[:, ::1] P):
        pass
