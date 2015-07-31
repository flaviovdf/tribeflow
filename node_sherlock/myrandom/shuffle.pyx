#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False

from __future__ import division, print_function

from node_sherlock.myrandom.random cimport rand

import numpy as np

cdef extern from 'stdlib.h':
    cdef void abort() nogil
    cdef void free(void *) nogil
    cdef void *malloc(size_t) nogil

cdef int partition(double *data, int bg, int ed) nogil:
    
    cdef double pivot_val = data[ed - 1]
    cdef int left = bg
    cdef int right = ed - 2
    cdef double val = 0
    while left < right:
        val = data[left]
        if val < pivot_val:
            left += 1
        else:
            data[left] = data[right]
            data[right] = val
            right -= 1
    
    if data[left] > pivot_val:
        data[ed - 1] = data[left]
        data[left] = pivot_val
        return left
    else:
        data[ed - 1] = data[left + 1]
        data[left + 1] = pivot_val
        return left + 1

cdef void qsort(double *data, int bg, int ed) nogil:
    cdef int pivot_pos = 0
    if bg < ed - 1:
        pivot_pos = partition(data, bg, ed)
        qsort(data, bg, pivot_pos)
        qsort(data, pivot_pos + 1, ed)

def _qsort(double[::1] data):
    qsort(&data[0], 0, data.shape[0])

cdef void *random_choice(double *data, int dsize, double *buff, \
        int bsize) nogil:
    
    cdef double *copy = <double *> malloc(dsize * sizeof(double))
    if copy == NULL:
        abort()

    cdef int i = 0
    for i in xrange(dsize):
        copy[i] = data[i]

    i = 0
    cdef int elem = 0
    cdef int left = bsize
    cdef double r = 0
    cdef double aux = 0
    while left > 0 and i < dsize:
        r = rand()
        elem = i + <int> (r * (dsize - i))
        
        buff[i] = copy[elem]
        aux = copy[i]
        copy[i] = copy[elem]
        copy[elem] = aux

        i += 1
        left -= 1
    
    qsort(buff, 0, bsize)
    free(copy)

def _rchoice(double[::1] data, int n):
    cdef double[::1] rv = np.zeros(n, dtype='d', order='C')
    random_choice(&data[0], data.shape[0], &rv[0], n)
    return np.asarray(rv)
