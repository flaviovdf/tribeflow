#-*- coding: utf8
from __future__ import division, print_function

from node_sherlock.mycollections.stamp_lists import StampLists
from node_sherlock.kernels import TStudentKernel
from numpy.testing import assert_almost_equal
from scipy.stats.distributions import t

import numpy as np
import node_sherlock

def test_pdf():
    for mu in [0, 1, 10]:
        for v in [1, 10]:
            for std in [1, 10]:
                priors = np.array([mu, v, std], dtype='d')
                kernel = TStudentKernel()
                kernel.build(999, 1, priors) #99... is just the max freed
                
                truth = t(v, loc=mu, scale=std)
                for x in np.linspace(-100, 100, 200):
                    print(mu, v, std, x, truth.pdf(x), kernel._pdf(x, 0, StampLists(1)))
                    assert_almost_equal(truth.pdf(x), \
                            kernel._pdf(x, 0, StampLists(1)))
