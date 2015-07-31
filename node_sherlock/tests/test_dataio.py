#-*- coding: utf8
'''
Unit tests for the main sherlock model.
'''
from __future__ import division, print_function

from node_sherlock.tests import files

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from node_sherlock import dataio

def test_initialize():
    tstamps, Trace, previous_stamps, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10)
    
    assert_equal(len(hyper2id), 3)
    assert_equal(len(source2id), 5) 
    assert_equal(len(dest2id), 6)
    
    assert_equal(Trace.shape[0], 10)
    assert_equal(Trace.shape[1], 4)
    
    for z in [0, 1]:
        assert previous_stamps._size(z) > 0

    assert_equal(count_h[0], 4)
    assert_equal(count_h[1], 4)
    assert_equal(count_h[2], 2)
    
    assert_equal(count_z.sum(), 10)
    
    #We can only test shapes and sum, since assignments are random
    assert_equal(Count_zh.shape, (2, 3))
    assert_equal(Count_sz.shape, (5, 2))
    assert_equal(Count_dz.shape, (6, 2))
    
    assert_equal(Count_zh.sum(), 10)
    assert_equal(Count_sz.sum(), 10)
    assert_equal(Count_dz.sum(), 10)
    
    assert (prob_topics_aux == 0).all()

    #Simple sanity check on topic assigmnets. Check if topics have valid
    #ids and if count matches count matrix        
    from collections import Counter
    c = Counter(Trace[:, -1])
    for topic in c:
        assert topic in [0, 1]
        assert c[topic] == count_z[topic]


def test_initialize_limits():
    tstamps, Trace, previous_stamps, Count_zh, Count_sz, Count_dz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            Psi_dz, hyper2id, source2id, dest2id = \
            dataio.initialize_trace(files.SIZE10, 2, 10, 2, 5)
    
    assert len(tstamps) == 3
