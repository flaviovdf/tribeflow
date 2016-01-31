'''
This simple example will run TribeFlow *serially*. No MPI based code is 
necessary.
'''
from __future__ import division, print_function

from tribeflow import dataio
from tribeflow import kernels
from tribeflow import fit

import pandas as pd
import numpy as np

in_fpath = 'example/lastfm_our.dat'
size = sum(1 for _ in open(in_fpath))
nz = 50

#Intervent kernel with eccdf heuristic. Priors are for the beta distribution.
inter_event_kernel =  kernels.names['eccdf']()
priors = np.array([1.0, nz - 1])
inter_event_kernel.build(size, nz, priors)

#Trace, num envs, alpha prior, beta prior, kernel, kernels priors as array
#num iter, burn in
rv = fit(in_fpath, nz, nz / 10, .001, inter_event_kernel, priors, 800, 300)

out_fpath = 'example/model.h5'
dataio.save_model(out_fpath, rv)

#Let's load the model so that we can get a feel of how to work with the pandas
#output
model = pd.HDFStore(out_fpath, 'r')

#Let's print the top 5 objects for each env.
Psi_sz = model["Psi_sz"].values
id2obj = dict((v, k) for k, v in model["source2id"].values)

for z in xrange(nz):
    top = Psi_sz[:, z].argsort()[::-1][:5]
    
    print(z)
    for i in xrange(5):
        print(id2obj[top[i]])
    print()
