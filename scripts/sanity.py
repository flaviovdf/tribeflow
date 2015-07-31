from __future__ import division, print_function

from node_sherlock import dataio
from node_sherlock import plearn

from node_sherlock.kernels import NullKernel
from node_sherlock.kernels import HawkesExponKernel
from node_sherlock.kernels import PPKernel
from node_sherlock.kernels import SFPKernel
from node_sherlock.kernels import SpikeMKernel

from node_sherlock.msteps.ga import GradientAscent

import pandas as pd
import numpy as np

#FPATH = './node_sherlock/tests/sample_data/delicious.dat'
FPATH = './node_sherlock/tests/sample_data/mlens.dat'

nz = 40
mstep = GradientAscent(nz, 0.001)
kernel = HawkesExponKernel(nz)
#kernel = PPKernel()
#kernel = SpikeMKernel()
#kernel = SFPKernel()
#kernel = NullKernel()

rv = plearn.fit(FPATH, nz, 50 / nz, .001, .001, 800, mstep, kernel, 4)

out_fpath = './sanity/model.h5'
dataio.save_model(out_fpath, rv)

Psi_sz = rv["Psi_sz"]
Psi_dz = rv["Psi_dz"]
TimeHp = rv["TimeHp"]

from collections import Counter
print(Counter(rv['assign']))

id2dest = dict((v, k) for k, v in rv["dest2id"].items())
id2source = dict((v, k) for k, v in rv["source2id"].items())

k = 20
for z in xrange(nz):
    print(TimeHp[z])
    top_source = Psi_sz[:, z].argsort()[::-1][:k]
    top_dest = Psi_dz[:, z].argsort()[::-1][:k]
    
    print(z)
    for i in xrange(k):
        print(id2source[top_source[i]])
    print()

    for i in xrange(k):
        print(id2dest[top_dest[i]])
    print()
    print()
    print()
