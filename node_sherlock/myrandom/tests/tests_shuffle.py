#-*- coding: utf8
from __future__ import division, print_function

from node_sherlock.myrandom.shuffle import _qsort
from node_sherlock.myrandom.shuffle import _rchoice

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

import numpy as np

def test_sort():
    data = np.array([], dtype='d', order='C')
    _qsort(data)
    assert_array_equal([], data)

    data = np.array([1], dtype='d', order='C')
    _qsort(data)
    assert_array_equal([1], data)
    
    data = np.array([2, 1], dtype='d', order='C')
    _qsort(data)
    assert_array_equal([1, 2], data)
    
    data = np.array([2, 2, 0, 1], dtype='d', order='C')
    _qsort(data)
    assert_array_equal([0, 1, 2, 2], data)
    
    data = np.array([0, 2, 2, 1], dtype='d', order='C')
    _qsort(data)
    assert_array_equal([0, 1, 2, 2], data)
    
    data = np.array([2, 0, 1, 2], dtype='d', order='C')
    _qsort(data)
    assert_array_equal([0, 1, 2, 2], data)
    
    correct = np.arange(100)
    data = np.asarray(correct, dtype='d', order='C')
    _qsort(data)
    assert_array_equal(correct, data)
    
    correct = np.arange(100)
    data = np.asarray(correct.copy()[::-1], dtype='d', order='C')
    _qsort(data)
    assert_array_equal(correct, data)
    
    correct = np.arange(100)
    data = np.asarray(correct.copy(), dtype='d', order='C')
    np.random.shuffle(data)
    _qsort(data)
    assert_array_equal(correct, data)

def test_rchoice():
    data = np.array([], dtype='d', order='C')
    rv = _rchoice(data, 0)
    assert_equal((0, ), rv.shape)
    
    data = np.array([3, 2, 1], dtype='d', order='C')
    rv = _rchoice(data, 3)
    assert_array_equal([1, 2, 3], rv)
    
    data = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='d', order='C')
    rv = _rchoice(data, 4)
    assert_array_equal(4, rv.shape[0])
    assert rv[1] > rv[0]
    assert rv[2] > rv[1]
    assert rv[3] > rv[2]
    
    for i in xrange(rv.shape[0]):
        assert rv[i] <= 8
