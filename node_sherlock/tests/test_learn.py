#-*- coding: utf8
'''
Unit tests for the main sherlock model.
'''
from __future__ import division, print_function

from node_sherlock.tests import files
from node_sherlock.kernels import NoopKernel

from numpy.testing import assert_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

from node_sherlock import dataio
from node_sherlock import _learn

import numpy as np

def test_posterior():
    assert_equal(.6086956521739131, _learn._dir_posterior(2, 3, 2, 0.8))

def test_sample():
    tstamps, Trace, previous_stamps, Count_zh, Count_sz, count_h, count_z, \
            prob_topics_aux, Theta_zh, Psi_sz, hyper2id, source2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
    kernel = NoopKernel()
    kernel.build(Trace.shape[0], Count_zh.shape[1], np.zeros(0, dtype='d'))

    tstamp_idx = 3
    hyper = Trace[tstamp_idx, 0]
    source = Trace[tstamp_idx, 1]
    dest = Trace[tstamp_idx, 2]
    old_topic = Trace[tstamp_idx, 3]

    new_topic = _learn._sample(tstamp_idx, tstamps, Trace, \
            previous_stamps, Count_zh, Count_sz, count_h, \
            count_z, .1, .1, prob_topics_aux, kernel)
    
    assert new_topic <= 3

def test_estep():
    tstamps, Trace, previous_stamps, Count_zh, Count_sz, count_h, count_z, \
            prob_topics_aux, Theta_zh, Psi_sz, hyper2id, source2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
    kernel = NoopKernel()
    kernel.build(Trace.shape[0], Count_zh.shape[1], np.zeros(0, dtype='d'))
    
    alpha_zh = .1
    beta_zs = .1

    assert_equal(Count_zh.sum(), 10)
    assert_equal(Count_sz.sum(), 20)
    
    assert_equal(count_h[0], 4)
    assert_equal(count_h[1], 4)
    assert_equal(count_h[2], 2)
    
    new_state = _learn._e_step(tstamps, Trace, previous_stamps, Count_zh, \
            Count_sz, count_h, count_z, alpha_zh, beta_zs, prob_topics_aux, \
            kernel)

    assert_equal(count_h[0], 4)
    assert_equal(count_h[1], 4)
    assert_equal(count_h[2], 2)
    
    assert_equal(Count_zh.sum(), 10)
    assert_equal(Count_sz.sum(), 20)

def test_em():
    
    tstamps, Trace, previous_stamps, Count_zh, Count_sz, count_h, count_z, \
            prob_topics_aux, Theta_zh, Psi_sz, hyper2id, source2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
    kernel = NoopKernel()
    kernel.build(Trace.shape[0], Count_zh.shape[1], np.zeros(0, dtype='d'))
    
    alpha_zh = .1
    beta_zs = .1
    
    assert (Theta_zh == 0).all()
    assert (Psi_sz == 0).all()
    
    old_Count_zh = Count_zh.copy()
    old_Count_sz = Count_sz.copy()
    old_count_h = count_h.copy()
    old_count_z = count_z.copy()

    _learn.em(tstamps, Trace, previous_stamps, Count_zh, Count_sz, count_h, \
            count_z, alpha_zh, beta_zs, prob_topics_aux, Theta_zh, Psi_sz, \
            10, 2, kernel)
    
    assert (Theta_zh > 0).sum() > 0
    assert (Psi_sz > 0).sum() > 0
    
    assert_almost_equal(1, Theta_zh.sum(axis=0))
    assert_almost_equal(1, Psi_sz.sum(axis=0))

    assert (old_Count_zh != Count_zh).any()
    assert (old_Count_sz != Count_sz).any()
    
    assert (old_count_h == count_h).all() #the count_h should not change
    assert (old_count_z != count_z).any()
