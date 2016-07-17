#-*- coding: utf8
from __future__ import division, print_function

import arrow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plac
import seaborn as sns
import toyplot
import toyplot.pdf
import toyplot.html

def main(model, from_, to):
    from_ = int(from_)
    to = int(to)

    store = pd.HDFStore(model)
    trace_fpath = store['trace_fpath'][0][0]
    Theta_zh = store['Theta_zh'].values
    hyper2id = dict(store['hyper2id'].values)
    id2hyper = dict((v, k) for k, v in hyper2id.items())

    aux = Theta_zh[from_] * Theta_zh[to]
    for i in aux.argsort()[::-1][:10]:
        print(id2hyper[i], 'f', Theta_zh[from_, i], 't', Theta_zh[to, i])

    store.close()
    
plac.call(main)
