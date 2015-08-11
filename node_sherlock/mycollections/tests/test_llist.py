#-*- coding: utf8
'''
Unit tests for the main sherlock model.
'''
from __future__ import division, print_function

from node_sherlock.mycollections.llist import MarkovianMemory

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

import numpy as np

def test_insert():
    mm = MarkovianMemory(4)
    assert_equal(0, mm._size(0))
    assert_equal(0, mm._size(1))
    assert_equal(0, mm._size(2))
    assert_equal(0, mm._size(3))

    mm._append_last(0, 1, 2.0)
    mm._append_last(0, 2, 3.0)
    mm._append_last(2, 2, 1.0)

    assert_equal(2, mm._size(0))
    assert_equal(0, mm._size(1))
    assert_equal(1, mm._size(2))
    assert_equal(0, mm._size(3))

    l = mm._to_list(0)
    assert_array_equal([(1, 2.0), (2, 3.0)], l)
    l = mm._to_list(1)
    assert_array_equal([], l)
    l = mm._to_list(2)
    assert_array_equal([(2, 1.0)], l)
