#-*- coding: utf8
from __future__ import division, print_function

from tribeflow.kernels import ECCDFKernel
from tribeflow.mycollections.stamp_lists import StampLists
from numpy.testing import assert_array_equal

import numpy as np
import tribeflow

def test_all():
    
    slists = StampLists(4)
    E = []
    for i in xrange(4):
        E.append([])
        for _ in xrange(200):
            e = np.random.rand()
            slists._append(i, e)
            E[i].append(e)

    kern = ECCDFKernel(True)
    kern.build(800, 4, np.array([1.0, 3.0]))

    kern._mstep(slists)

    for i in xrange(4):
        assert_array_equal(sorted(E[i]), slists._get_all(i))


    for i in xrange(4):
        for _ in xrange(200):
            p = kern._pdf(i, np.random.rand(), slists)
            assert p <= 1
            assert p >= 0
