#-*- coding: utf8
from __future__ import division, print_function

from tribeflow.sorting.introsort import _sort

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

import numpy as np

def test__sort():
    data = np.array([], dtype='d', order='C')
    _sort(data)
    assert_array_equal([], data)

    data = np.array([1], dtype='d', order='C')
    _sort(data)
    assert_array_equal([1], data)
    
    data = np.array([2, 1], dtype='d', order='C')
    _sort(data)
    assert_array_equal([1, 2], data)
    
    data = np.array([2, 2, 0, 1], dtype='d', order='C')
    _sort(data)
    assert_array_equal([0, 1, 2, 2], data)
    
    data = np.array([0, 2, 2, 1], dtype='d', order='C')
    _sort(data)
    assert_array_equal([0, 1, 2, 2], data)
    
    data = np.array([2, 0, 1, 2], dtype='d', order='C')
    _sort(data)
    assert_array_equal([0, 1, 2, 2], data)
    
    correct = np.arange(100)
    data = np.asarray(correct, dtype='d', order='C')
    _sort(data)
    assert_array_equal(correct, data)
    
    correct = np.arange(100)
    data = np.asarray(correct.copy()[::-1], dtype='d', order='C')
    _sort(data)
    assert_array_equal(correct, data)
    
    correct = np.arange(100)
    data = np.asarray(correct.copy(), dtype='d', order='C')
    np.random.shuffle(data)
    _sort(data)
    assert_array_equal(correct, data)
