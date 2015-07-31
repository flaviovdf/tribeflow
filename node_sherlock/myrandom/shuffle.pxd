#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False

from __future__ import division, print_function

cdef int partition(double *data, int bg, int ed) nogil
cdef void qsort(double *data, int bg, int ed) nogil
cdef void *random_choice(double *data, int dsize, double *buff, \
        int bsize) nogil
