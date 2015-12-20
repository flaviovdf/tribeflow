#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False

from __future__ import division, print_function

from cython cimport view
import numpy as np

cdef extern from 'stdio.h':
    cdef void abort() nogil
    cdef void *malloc(size_t) nogil
    cdef void *realloc(void *, size_t) nogil
    cdef void free(void *) nogil

cdef class StampLists:
    
    def __cinit__(self, int num_topics, int increment=5000):
        self.values = <double **> malloc(num_topics * sizeof(double *))
        if self.values == NULL:
            abort()

        self.curr_sizes = view.array(shape=(num_topics, ), \
                itemsize=sizeof(int), format='i')
        self.limits = view.array(shape=(num_topics, ), \
                itemsize=sizeof(int), format='i')

        cdef int z
        for z in xrange(num_topics):
            self.curr_sizes[z] = 0
            self.limits[z] = increment
            self.values[z] = <double *> malloc(increment * sizeof(double))

            if self.values[z] == NULL:
                abort()
        
        self.inc = increment
    
    def copy(self):
        cdef StampLists rv = StampLists(self.curr_sizes.shape[0], self.inc)
        cdef int z
        cdef int i
        for z in xrange(self.limits.shape[0]):
            rv.values[z] = <double *> realloc(rv.values[z], \
                    self.limits[z] * sizeof(double))
            if rv.values[z] == NULL:
                abort()
            
            for i in xrange(self.curr_sizes[z]):
                rv.values[z][i] = self.values[z][i]

        rv.limits[:] = self.limits
        rv.curr_sizes[:] = self.curr_sizes
        return rv

    cdef void append(self, int topic, double value) nogil:
        if self.curr_sizes[topic] == self.limits[topic]:
            self.values[topic] = <double *> realloc(self.values[topic], \
                    (self.inc + self.limits[topic]) * sizeof(double))
            
            if self.values[topic] == NULL:
                abort()

            self.limits[topic] = self.inc + self.limits[topic]
        
        self.values[topic][self.curr_sizes[topic]] = value
        self.curr_sizes[topic] += 1
    
    def _append(self, int topic, double value):
        self.append(topic, value)
    
    def _extend(self, int topic, double[:] values):
        cdef int i
        with nogil:
            for i in range(values.shape[0]):
                self.append(topic, values[i])

    cdef double get(self, int topic, int idx) nogil:
        if idx >= self.curr_sizes[topic]:
            return -1
        
        return self.values[topic][idx]
    
    def _get(self, int topic, int idx):
        return self.get(topic, idx)
    
    cdef double *get_all(self, int topic) nogil:
        return self.values[topic]

    def _get_all(self, int topic):
        
        cdef int t_size = self.curr_sizes[topic]
        cdef double[::1] rv = np.zeros(t_size) 
        
        cdef int i
        with nogil:
            for i in range(t_size):
                rv[i] = self.values[topic][i]
        return rv

    cdef int size(self, int topic) nogil:
        return self.curr_sizes[topic]
    
    def _size(self, int topic):
        return self.size(topic)

    cdef void clear(self) nogil:
        cdef int i
        for i in xrange(self.curr_sizes.shape[0]):
            self.curr_sizes[i] = 0
    
    def _clear_one(self, int topic):
        self.curr_sizes[topic] = 0

    def _clear(self):
        self.clear()

    def __dealloc__(self):
        self.clear()
        if self.values != NULL:
            for i in xrange(self.curr_sizes.shape[0]):
                free(self.values[i])
            free(self.values)
