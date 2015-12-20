#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False

from __future__ import print_function, division

from tribeflow.mycollections.stamp_lists cimport StampLists

cdef class Kernel:
    
    cdef double pdf(self, double x, int z, StampLists stamps) nogil

    cdef void mstep(self, StampLists stamps) nogil
