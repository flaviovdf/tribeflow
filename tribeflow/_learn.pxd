#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False
from __future__ import print_function, division

cdef inline double dir_posterior(double joint_count, double global_count, \
        double num_occurences, double smooth) nogil
