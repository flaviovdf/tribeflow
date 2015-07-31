#-*- coding: utf8
'''Kernels are placed here. Mostly written in Cython code'''
from __future__ import division, print_function

from node_sherlock.kernels.eccdf import ECCDFKernel
from node_sherlock.kernels.noop import NoopKernel
from node_sherlock.kernels.tstudent import TStudentKernel

names = {'eccdf': ECCDFKernel, \
        'noop': NoopKernel,
        'tstudent': TStudentKernel}
