#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False

from __future__ import division, print_function

cdef extern from 'stdio.h':
    cdef void abort() nogil
    cdef void *malloc(size_t) nogil
    cdef void free(void *) nogil
    cdef int printf(char *, ...) nogil

cdef class MarkovianMemory:
    
    def __cinit__(self, int num_keys):
        self.first_per_key = <Node **> malloc(num_keys * sizeof(Node *))
        self.last_per_key = <Node **> malloc(num_keys * sizeof(Node *))
        self.sizes = <int *> malloc(num_keys * sizeof(int))
        self.num_keys = num_keys

        if self.first_per_key == NULL or self.last_per_key == NULL or \
                self.sizes == NULL:
            abort()

        cdef int i = 0
        for i in xrange(num_keys):
            self.first_per_key[i] = NULL
            self.last_per_key[i] = NULL
            self.sizes[i] = 0

    cdef void append_last(self, int key, int obj, double dt) nogil:
        
        cdef int size = self.sizes[key]
        
        cdef Node *new_node = <Node *> malloc(sizeof(Node))
        if new_node == NULL:
            abort()

        if size == 0:
            self.first_per_key[key] = new_node
            self.last_per_key[key] = new_node
        else:
            self.last_per_key[key].next_node = new_node
            self.last_per_key[key] = new_node
        
        new_node.obj = obj
        new_node.dt = dt
        new_node.next_node = NULL

        self.sizes[key] = size + 1
    
    def _append_last(self, int key, int obj, double dt):
        self.append_last(key, obj, dt)

    cdef void remove_first(self, int key) nogil:
        cdef int size = self.sizes[key]
        
        if size == 0:
            return
        
        cdef Node *to_free = self.first_per_key[key]
        self.first_per_key[key] = to_free.next_node
        free(to_free)
        
        self.sizes[key] = size - 1
        if self.sizes[key] == 0:
            self.first_per_key[key] = NULL
            self.last_per_key[key] = NULL
    
    def _remove_first(self, int key):
        self.remove_first(key)

    cdef Node *get_first(self, int key) nogil:
        return self.first_per_key[key]
    
    def _to_list(self, int key):
        cdef list rv = []
        cdef Node *next_node = self.get_first(key)
        while next_node != NULL:
            rv.append((next_node.obj, next_node.dt))
            next_node = next_node.next_node
        return rv
    
    cdef int size(self, int key) nogil:
        return self.sizes[key]
    
    def _size(self, int key):
        return self.size(key)

    cdef void clear(self) nogil:
        cdef Node *to_free = NULL
        cdef Node *next_node = NULL
        cdef int key = 0
        
        for key in xrange(self.num_keys):
            to_free = self.first_per_key[key]
            while to_free != NULL:
                next_node = to_free.next_node
                free(to_free)
                to_free = next_node

            self.first_per_key[key] = NULL
            self.last_per_key[key] = NULL
            self.sizes[key] = 0
    
    def _clear(self):
        self.clear()

    def __dealloc__(self):
        self.clear()
        
        if self.first_per_key != NULL:
            free(self.first_per_key)
        
        if self.last_per_key != NULL:
            free(self.last_per_key)
        
        if self.sizes != NULL:
            free(self.sizes)
