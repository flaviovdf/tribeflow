#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False

from __future__ import division, print_function

cdef class StampLists:

    cdef int[:] limits
    cdef int[:] curr_sizes
    cdef int inc
    cdef double **values

    cdef void append(self, int topic, double value) nogil
    cdef double get(self, int topic, int idx) nogil
    cdef double *get_all(self, int topic) nogil
    cdef int size(self, int topic) nogil
    cdef void clear(self) nogil
