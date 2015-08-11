#-*- coding: utf8
from __future__ import division, print_function

from node_sherlock.sorting.binsearch import _bsp

import numpy as np

def test_bsp():
    x = np.zeros(shape=(0, ), dtype='d')
    p = _bsp(x, 1)
    assert p == 0
    
    p = _bsp(x, 0)
    assert p == 0
    
    p = _bsp(x, -1)
    assert p == 0

    x = np.zeros(shape=(1, ), dtype='d')
    x[0] = 10
    p = _bsp(x, 1)
    assert p == 0
    
    p = _bsp(x, 0)
    assert p == 0
    
    p = _bsp(x, -1)
    assert p == 0
    
    p = _bsp(x, 10)
    assert p == 0
    
    p = _bsp(x, 11)
    assert p == 1
    
    x = np.zeros(shape=(2, ), dtype='d')
    x[0] = 2
    x[1] = 8
    p = _bsp(x, 1)
    assert p == 0
    
    p = _bsp(x, 0)
    assert p == 0
    
    p = _bsp(x, -1)
    assert p == 0
    
    p = _bsp(x, 5)
    assert p == 1
    
    p = _bsp(x, 10)
    assert p == 2
    
    p = _bsp(x, 11)
    assert p == 2
