#-*- coding: utf8
'''
Unit tests for the main sherlock model.
'''
from __future__ import division, print_function

from tribeflow.mycollections import stamp_lists

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

import numpy as np

def test_one():
    l = stamp_lists.StampLists(1, 10)
    for i in xrange(1000):
        l._append(0, i + 10)

    assert_equal(1000, l._size(0))
    for i in xrange(1000):
        assert_equal(l._get(0, i), i + 10)
    
    l._clear()
    assert_equal(0, l._size(0))
    for i in xrange(120):
        l._append(0, i + 8)

    assert_equal(120, l._size(0))
    for i in xrange(120):
        assert_equal(l._get(0, i), i + 8)
    
    l._clear()
    l._clear()

def test_many():
    num_topics = 10
    l = stamp_lists.StampLists(num_topics, 1)
    for k in xrange(num_topics):
        for i in xrange(k + 1):
            l._append(k, i + k)

    for k in xrange(num_topics):
        assert_equal(k + 1, l._size(k))

    l._clear()
    for k in xrange(num_topics):
        assert_equal(0, l._size(k))
    
    for k in xrange(num_topics):
        for i in xrange(120):
            l._append(k, i + k)

        assert_equal(120, l._size(k))
        for i in xrange(120):
            assert_equal(l._get(k, i), i + k)

def test_get_all():
    num_topics = 10
    l = stamp_lists.StampLists(num_topics, 1)
    l._append(1, 1)
    l._append(1, 2)
    
    assert_array_equal([1, 2], np.array(l._get_all(1)))
    assert_array_equal([], np.array(l._get_all(5)))

def test_copy():
    num_topics = 2
    l = stamp_lists.StampLists(num_topics, 1)
    l._append(0, 1)
    l._append(0, 2)
    l._append(1, 1)

    c = l.copy()
    assert_array_equal([1, 2], np.array(c._get_all(0)))
    assert_array_equal([1], np.array(c._get_all(1)))

    l._append(0, 1)
    l._append(1, 2)
    
    assert_array_equal([1, 2], np.array(c._get_all(0)))
    assert_array_equal([1], np.array(c._get_all(1)))
    
    c._append(0, 1)
    c._append(1, 2)
    
    assert_array_equal([1, 2, 1], np.array(c._get_all(0)))
    assert_array_equal([1, 2], np.array(c._get_all(1)))
