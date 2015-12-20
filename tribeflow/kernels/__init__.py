#-*- coding: utf8
'''Kernels are placed here. Mostly written in Cython code'''
from __future__ import division, print_function

from tribeflow.kernels.eccdf import ECCDFKernel
from tribeflow.kernels.noop import NoopKernel
from tribeflow.kernels.tstudent import TStudentKernel

names = {'eccdf': ECCDFKernel, \
        'noop': NoopKernel,
        'tstudent': TStudentKernel}
