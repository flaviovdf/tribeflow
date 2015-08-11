#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False

from __future__ import print_function, division

from node_sherlock.mycollections.stamp_lists cimport StampLists
from node_sherlock.kernels.base cimport Kernel

import numpy as np

cdef double PI = 3.14159265359
cdef int MAX_FREE = 71

cdef extern from 'math.h':
    double sqrt(double) nogil

cdef extern from 'stdio.h':
    int printf(char *, ...) nogil

cdef class TStudentKernel(Kernel):
    
    cdef double[::1] cte
    cdef double[::1] priors
    cdef double[:, ::1] P

    def build(self, int trace_size, int nz, double[::1] priors):
        
        assert priors.shape[0] == 3
        cdef int n = trace_size 
        cdef double[::1] cte = np.zeros(MAX_FREE, dtype='d')
        
        cte[0] = 0        
        if trace_size >= 1:
            cte[1] = 1.0 / PI
        if trace_size >= 2:
            cte[2] = 1.0 / (2.0 * np.sqrt(2))
        if trace_size >= 3:
            cte[3] = 2.0 / (np.sqrt(3) * PI)
        if trace_size >= 4:
            cte[4] = 3.0 / 8.0
        if trace_size >= 5:
            cte[5] = 8.0 / (3.0 * np.sqrt(5) * PI)
        
        cdef double even = 4.0 * 2.0
        cdef double odd = 5.0 * 3.0
        cdef int i
        for i in xrange(6, MAX_FREE): 
            if i % 2 == 0:
                cte[i] = odd / (even * 2 * np.sqrt(i))
                even = even * i
            else:
                cte[i] = even / (odd * PI * np.sqrt(i))
                odd = odd * i
        
        self.cte = cte
        self.P = np.zeros(shape=(nz, 3), dtype='d')
        for i in xrange(nz):
            self.P[i, 0] = priors[0]
            self.P[i, 1] = priors[1]
            self.P[i, 2] = priors[2]
        self.priors = np.array(priors, copy=True)

    def get_priors(self):
        return np.array(self.priors)

    def get_state(self):
        return np.array(self.P)

    def update_state(self, double[:, ::1] P):
        assert P.shape[0] == self.P.shape[0]
        assert P.shape[1] == self.P.shape[1]
        self.P = P

    cdef double pdf(self, double x, int z, StampLists stamps) nogil:
        
        cdef double mu = self.P[z, 0]
        cdef double v = self.P[z, 1]
        cdef double sigma = self.P[z, 2]
        cdef double c = 1.0

        cdef int free = <int>v
        if free <= 0 or sigma == 0:
            return 0.0
        elif free < self.cte.shape[0]:
            c = self.cte[free] / sigma
        else:
            c = self.cte[self.cte.shape[0] - 1] / sigma

        return c * (1 + (1 / v) * (((x - mu) / sigma) ** 2)) ** -((v + 1) / 2)
    
    def _pdf(self, x, z, stamps):
        return self.pdf(x, z, stamps)

    cdef void mstep(self, StampLists stamps) nogil:
        
        cdef int nz = self.P.shape[0]
        
        cdef double mu0 = self.priors[0]
        cdef double v0 = self.priors[1]
        cdef double sigma0 = self.priors[2]
        cdef double n0 = 1.0
        
        cdef double *obs
        cdef int n
        cdef double mean
        cdef double ssq
        cdef int z
        cdef int i
        for z in xrange(nz):
            n = stamps.size(z)
            if n == 0:
                self.P[z, 0] = mu0
                self.P[z, 1] = v0
                self.P[z, 2] = sigma0
                break

            obs = stamps.get_all(z)
            mean = 0.0
            ssq = 0.0

            for i in xrange(n):
                mean += obs[i]
            mean = mean / n
            for i in xrange(n):
                ssq += (obs[i] - mean) ** 2

            self.P[z, 0] = (n0 * mu0 + n * mean) / (n0 + n)
            self.P[z, 1] = (n + v0)
            self.P[z, 2] = sqrt(\
                    (v0 * (sigma0 ** 2) + (n - 1) * ssq + \
                        ((n0 * n * ((mean - mu0) ** 2)) / (n0 + n))) \
                    / ((v0 + n) * (n0 + n)))
