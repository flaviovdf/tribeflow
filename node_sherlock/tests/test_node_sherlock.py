#-*- coding: utf8
'''
Unit tests for the main sherlock model.
'''
from __future__ import division, print_function

from node_sherlock import dataio
from node_sherlock import learn

from node_sherlock.tests import files
from node_sherlock.kernels import NoopKernel

from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal

import numpy as np

def test_full_learn_null():
    kernel = NoopKernel()
    kernel.build(1, 20, np.zeros(0, dtype='d'))
    rv = learn.fit(files.SIZE1K, 20, .1, .1, .1, kernel, \
            np.zeros(0, dtype='d'), 1000, 200)
    
    Count_zh = rv['Count_zh']
    Count_sz = rv['Count_sz'] 
    Count_dz = rv['Count_dz'] 
    
    assert_equal(Count_zh.sum(), 10000)
    assert_equal(Count_sz.sum(), 10000)
    assert_equal(Count_dz.sum(), 10000)
    
    count_h = rv['count_h']
    count_z = rv['count_z']

    assert_equal(count_h.sum(), 10000)
    assert_equal(count_z.sum(), 10000)

    assert rv['assign'].shape == (10000, )

    Theta_zh = rv['Theta_zh']
    Psi_sz = rv['Psi_sz']
    Psi_dz = rv['Psi_dz']
    
    assert_almost_equal(1, Theta_zh.sum(axis=0))
    assert_almost_equal(1, Psi_sz.sum(axis=0))
    assert_almost_equal(1, Psi_dz.sum(axis=0))
