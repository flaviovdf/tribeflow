#-*- coding: utf8
'''Main code. Entry point for tribeflow from the command line. Just run python
main.py and check out the docs'''
from __future__ import division, print_function

from tribeflow import dataio
from tribeflow import dynamic
from tribeflow import learn
from tribeflow import kernels
from tribeflow import plearn

from mpi4py import MPI

import argparse
import multiprocessing
import numpy as np
import os
import time

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('trace_fpath', help='The trace to learn topics from', \
            type=str)
    parser.add_argument('num_topics', help='The number of topics to learn', \
            type=int)
    parser.add_argument('model_fpath', \
            help='The name of the model file (a h5 file)', type=str)

    parser.add_argument('--num_iter', help='The number of iterations', \
            type=int, default=1000)
    parser.add_argument('--burn_in', \
            help='The burn in (ignored if using mpi)', type=int, \
            default=300)
    
    parser.add_argument('--dynamic', \
            help='If we should employ the dynamic strategy', \
            type=bool, default=False)
    parser.add_argument('--num_batches', \
            help='Number of batches in dynamic case', type=int, default=10)
    
    parser.add_argument('--alpha_zh', \
            help='Value of alpha_zh (alpha) hyper. Defaults to 50 / nz', \
            type=float, default=None)
    parser.add_argument('--beta_zs', help='Value of beta_zs (beta) hyper', \
            type=float, default=0.001)
    
    parser.add_argument('--kernel', choices=kernels.names,
            help='The kernel to use', type=str, default='noop')
    parser.add_argument('--residency_priors', nargs='+',
            help='Priors for the residency time dist', type=float, default=None)
    
    parser.add_argument('--leaveout', \
            help='The number of transitions to leave for test', type=float, \
            default=0)
    
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size 
    try:
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size 
        single_thread = size <= 1
    except:
        single_thread = True
    
    if not single_thread and rank != plearn.MASTER:
        plearn.work()
    else:
        started = time.mktime(time.localtime())
        num_lines = 0
        with open(args.trace_fpath) as trace_file:
            num_lines = sum(1 for _ in trace_file)
        
        if args.leaveout > 0:
            leave_out = min(1, args.leaveout)
            if leave_out == 1:
                print('Leave out is 1 (100%), nothing todo')
                return
            from_ = 0
            to = int(num_lines - num_lines * leave_out)
        else:
            from_ = 0
            to = np.inf

        if args.alpha_zh is None:
            alpha_zh = 50.0 / args.num_topics
        else:
            alpha_zh = args.alpha_zh
        
        kernel = kernels.names[args.kernel]()
        if args.residency_priors:
            residency_priors = np.array(args.residency_priors, dtype='d')
        else:
            residency_priors = np.zeros(shape=(0, ), dtype='d')
        
        if np.isinf(to):
            kernel.build(num_lines - from_, args.num_topics, residency_priors)
        else:
            kernel.build(to - from_, args.num_topics, residency_priors)
        
        if single_thread:
            print('Not on MPI mode or just one MPI proc, running single thread')
            dyn = args.dynamic
            if dyn:
                num_iter = args.num_iter // args.num_batches
                rv = dynamic.fit(args.trace_fpath, args.num_topics, alpha_zh, \
                        args.beta_zs, kernel, residency_priors, \
                        num_iter, args.num_batches, False, from_=from_, to=to)
            else:
                rv = learn.fit(args.trace_fpath, args.num_topics, alpha_zh, \
                        args.beta_zs, kernel, residency_priors, \
                        args.num_iter, args.burn_in, from_=from_, to=to)
        else:
            dyn = args.dynamic
            if dyn:
                num_iter = args.num_iter // args.num_batches
                rv = dynamic.fit(args.trace_fpath, args.num_topics, alpha_zh, \
                        args.beta_zs, kernel, residency_priors, num_iter, \
                        args.num_batches, True, from_=from_, to=to)
            else:
                rv = plearn.fit(args.trace_fpath, args.num_topics, alpha_zh, \
                        args.beta_zs, kernel, residency_priors, args.num_iter, \
                        from_=from_, to=to)
        
        ended = time.mktime(time.localtime())
        rv['training_time'] = np.array([ended - started])
        dataio.save_model(args.model_fpath, rv)
        print('Learning took', ended - started, 'seconds')

if __name__ == '__main__':
    main()
