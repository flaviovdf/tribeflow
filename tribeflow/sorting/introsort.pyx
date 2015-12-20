#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False
#
# This code was copied from sklearn. Faster than my other sort.
# Our licenses are compatible.

from __future__ import division, print_function

cdef extern from 'math.h':
    double log(double) nogil

# Sort n-element arrays pointed to bydata 
# Algorithm: Introsort (Musser, SP&E, 1997).
cdef void sort(double *data, int n) nogil:
    cdef int maxd = 2 * <int>log(n)
    introsort(data, n, maxd)

def _sort(double[::1] data):
    sort(&data[0], data.shape[0])

cdef inline void swap(double *data, int i, int j) nogil:
    # Helper for sort
    data[i], data[j] = data[j], data[i]

cdef inline double median3(double *data, int n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef double a = data[0], b = data[n / 2], c = data[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b

# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef inline void introsort(double *data, int n, int maxd) nogil:
    cdef double pivot
    cdef int i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(data, n)
            return
        maxd -= 1

        pivot = median3(data, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if data[i] < pivot:
                swap(data, i, l)
                i += 1
                l += 1
            elif data[i] > pivot:
                r -= 1
                swap(data, i, r)
            else:
                i += 1

        introsort(data, l, maxd)
        data += r
        n -= r

cdef inline void sift_down(double *data, int start, int end) nogil:
    # Restore heap order indata[start:end] by moving the max element to start.
    cdef int child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and data[maxind] <data[child]:
            maxind = child
        if child + 1 < end and data[maxind] <data[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(data, root, maxind)
            root = maxind

cdef void heapsort(double *data, int n) nogil:
    cdef int start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(data, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(data, 0, end)
        sift_down(data, 0, end)
        end = end - 1
