#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False
from __future__ import division, print_function

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

def _bsp(double[::1] array, double value):
    return bsp(&array[0], value, array.shape[0])
