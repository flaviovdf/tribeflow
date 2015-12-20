#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False

from __future__ import division, print_function

cdef struct Node:
    int obj
    double dt
    Node *next_node

cdef class MarkovianMemory:
    
    cdef Node **first_per_key
    cdef Node **last_per_key
    cdef int *sizes
    cdef int num_keys

    cdef void append_last(self, int key, int obj, double dt) nogil
    cdef void remove_first(self, int key) nogil
    cdef Node *get_first(self, int key) nogil
    cdef int size(self, int key) nogil
    cdef void clear(self) nogil
