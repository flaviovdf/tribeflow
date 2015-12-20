#-*- coding: utf8
'''
Unit tests for the main sherlock model.
'''
from __future__ import division, print_function

from tribeflow import dataio
from tribeflow import learn

from tribeflow.tests import files
from tribeflow.kernels import NoopKernel

from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal

import numpy as np

def test_full_learn_null():
    kernel = NoopKernel()
    kernel.build(1, 20, np.zeros(0, dtype='d'))
    rv = learn.fit(files.SIZE1K, 20, .1, .1, kernel, \
            np.zeros(0, dtype='d'), 1000, 200)
    
    Count_zh = rv['Count_zh']
    Count_sz = rv['Count_sz'] 
    
    assert_equal(Count_zh.sum(), 1000)
    assert_equal(Count_sz.sum(), 2000) 
    
    count_h = rv['count_h']
    count_z = rv['count_z']

    assert_equal(count_h.sum(), 1000)
    assert_equal(count_z.sum(), 2000)

    assert rv['assign'].shape == (1000, )

    Theta_zh = rv['Theta_zh']
    Psi_sz = rv['Psi_sz']
    
    assert_almost_equal(1, Theta_zh.sum(axis=0))
    assert_almost_equal(1, Psi_sz.sum(axis=0))
