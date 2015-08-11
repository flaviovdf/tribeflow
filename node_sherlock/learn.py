#-*- coding: utf8
from __future__ import division, print_function

from node_sherlock import dataio

from _learn import em
from collections import OrderedDict

import numpy as np
import os

def prepare_results(trace_fpath, num_topics, alpha_zh, beta_zs, \
        kernel, residency_priors, num_iter, burn_in, tstamps, Trace, \
        Count_zh, Count_sz, count_h, count_z, \
        prob_topics_aux, Theta_zh, Psi_sz, hyper2id, source2id, \
        from_, to):
    
    rv = OrderedDict()
    
    rv['num_topics'] = np.asarray([num_topics])
    rv['trace_fpath'] = np.asarray([os.path.abspath(trace_fpath)])
    
    rv['alpha_zh'] = np.asarray([alpha_zh])
    rv['beta_zs'] = np.asarray([beta_zs])
    
    rv['num_iter'] = np.asarray([num_iter])
    rv['burn_in'] = np.asarray([burn_in])

    rv['from_'] = np.asarray([from_])
    rv['to'] = np.asarray([to])
    
    rv['tstamps'] = tstamps
    rv['Count_zh'] = Count_zh
    rv['Count_sz'] = Count_sz
    rv['count_h'] = count_h
    rv['count_z'] = count_z
    
    rv['Theta_zh'] = Theta_zh
    rv['Psi_sz'] = Psi_sz
    
    #TODO: very ugly sollution to save class name.
    kname = str(kernel.__class__).split("'")[1]
    rv['kernel_class'] = np.array([kname])
    rv['residency_priors'] = residency_priors
    rv['P'] = kernel.get_state()

    rv['assign'] = Trace[:, -1]
    rv['hyper2id'] = hyper2id
    rv['source2id'] = source2id
    return rv 

def fit(trace_fpath, num_topics, alpha_zh, beta_zs, kernel, \
        residency_priors, num_iter, burn_in, from_=0, to=np.inf):
    '''
    Learns the latent topics from a temporal hypergraph trace. 

    Node-Sherlock is a EM algorithm in which the E-Step is a gibbs sample update,
    thus the reason we use a `num_iter` and `burn_in` approach, instead of the 
    usual convergence approach.

    Parameters
    ----------
    trace_fpath : str
        The path of the trace. Each line should be a \
                (timestamp, hypernode, source, destination) where the \
                timestamp is a long (seconds or milliseconds from epoch).

    num_topics : int
        The number of latent spaces to learn

    alpha_zh : float
        The value of the alpha_zh hyperparameter

    beta_zs : float
        The value of the beta_zs (beta) hyperaparameter

    kernel : Kernel object
        The kernel to use

    residency_priors : array of float
        The kernel hyper parameters

    num_iter : int
        The number of iterations to learn the model from

    burn_in : int
        The burn_in of the chain
    
    Returns
    -------
    
    TODO: explain this better. For the time being, see the keys of the dict.
    A dictionary with the results.
    '''
    tstamps, Trace, previous_stamps, Count_zh, Count_sz, \
            count_h, count_z, prob_topics_aux, Theta_zh, Psi_sz, \
            hyper2id, source2id = \
            dataio.initialize_trace(trace_fpath, num_topics, num_iter, \
            from_, to)
    
    em(tstamps, Trace, previous_stamps, Count_zh, \
            Count_sz, count_h, count_z, alpha_zh, beta_zs, \
            prob_topics_aux, Theta_zh, Psi_sz, num_iter, \
            burn_in, kernel)
    
    rv = prepare_results(trace_fpath, num_topics, alpha_zh, beta_zs, \
            kernel, residency_priors, num_iter, burn_in, tstamps, Trace, \
            Count_zh, Count_sz, count_h, \
            count_z, prob_topics_aux, Theta_zh, Psi_sz, hyper2id, \
            source2id, from_, to)
    rv['algorithm'] = np.asarray(['serial gibbs + em'])
    return rv
