#-*- coding: utf8
from hyptrails.trial_roulette import *
from pathtools.markovchain import MarkovChain
from scipy.sparse import *

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from random import choice

def create_structure_trails(g, n, length, from_, to):
    i = n
    trails = []
    while(i>0):
        #pick random vertex of G
        v = choice(g.nodes()[from_:to])
        trail = []
        trail.append(v)
        #let us try to produce trails with length 5
        x = length -1 
        while x>0:
            #pick random neighbor node
            v = choice(g.neighbors(v))
            trail.append(v)
            x -= 1

        i-=1
        trails.append(trail)
    return trails

def create_selfloop_trails(g, n, length, from_, to):
    i = n
    trails = []
    while(i>0):
        #pick random vertex of G
        v = choice(g.nodes()[from_:to])
        trail = []
        trail.append(v)
        #let us try to produce trails with length 5
        x=length -1 
        while x>0:
            trail.append(v)
            x -= 1

        i-=1
        trails.append(trail)
    return trails

def create_uniform_trails(g, n, length, from_, to):
    i = n
    trails = []
    while(i>0):
        #pick random vertex of G
        v = choice(g.nodes()[from_:to])
        trail = []
        trail.append(v)
        #let us try to produce trails with length 5
        x=length -1
        while x>0:
            new = choice(g.nodes()[from_:to])
            if new != v:
                v = new
                trail.append(v)
                x -= 1

        i-=1
        trails.append(trail)
    return trails

def struc_prior(paths, vocab, g, ax):

    # We start by creating the hypothesis matrix G
    i_indices = list()
    j_indices = list()
    values = list()
    # Let us iterate through all nodes
    for k,v in vocab.iteritems():
        # Get the neighbors of each node (i.e., those nodes the current node links to)
        # and set the corresponding value of the hypothesis matrix G to 1
        for w in g.neighbors(k):
            i_indices.append(v)
            j_indices.append(vocab[w])
            values.append(1)

    shape_matrix = (max(vocab.itervalues()) + 1, max(vocab.itervalues()) + 1)
    
    # Next we iterate through some values of k (hypothesis weighting factor)
    evidences = {}
    shape = len(vocab)
    for i in xrange(5):
        # If k = 0 then we can simply use an empty matrix
        if i == 0:
            prior = csr_matrix((shape,shape), dtype=np.float64)
        # Otherwise, we need to build the hypothesis matrix and elicit the prior
        else:
            # Creating the matrix with above defined values
            matrix_struc = csr_matrix((values, (i_indices, j_indices)),
                             shape=shape_matrix, dtype=np.float64)

            # Here we elicit the Dirichlet prior from expressed hypothesis matrix
            # We use the row-wise (trial) roulette chip distribution (i.e., same amount of chips for each row)
            # chips = number of chips to distribute PER ROW
            # The HypTrails paper suggests to distribute |S|*k chips per row.
            # As we ignore self-loops, we distribute (|S|-1)*k chips per row.
            chips = i*(matrix_struc.shape[0]-1.)
            prior = distr_chips_row(matrix_struc, chips, n_jobs=1)

        # Now, we can pass everything to the Markov chain framework and calculate corresponding evidences
        # k=1 corresponds to the order of the MC (first-order)
        # reset=False can be set to true if we want to work with a reset state (start end end state for each trail)
        # prior=1. corresponds to the initial uniform prior that is necessary for ensuring proper priors
        # specific_prior is our elicited prior
        # specific_prior_vocab is the vocabulary mapping the indices of the prior to the states
        # state_count is the number of states; this ensures the correct number of states (suggested to always set)
        markov = MarkovChain(k=1, use_prior=True, reset = False, prior=1., specific_prior=prior,
                                specific_prior_vocab = vocab, state_count=len(vocab), modus="bayes")
        markov.prepare_data(paths)
        markov.fit(paths)

        evidence = markov.bayesian_evidence()
        evidences[i] = evidence
    
    ax.plot(evidences.keys(), evidences.values(), marker='o', clip_on = False, label="structural", linestyle='--')
    return evidences

def selfloop_prior(paths, vocab, ax):
    
    #The steps are similar to the structural hypothesis described above
    shape = len(vocab)
    evidences = {}
    for i in xrange(5):
        if i == 0:
            prior = csr_matrix((shape,shape), dtype=np.float64)

        else:
            matrix_selfloop = lil_matrix((shape,shape), dtype=np.float64)
            # As we only believe in self-loops we only believe in transitions from the diagonal of Q
            matrix_selfloop.setdiag(1)
            matrix_selfloop = matrix_selfloop.tocsr()

            chips = i*(matrix_selfloop.shape[0]-1.)
            prior = distr_chips_row(matrix_selfloop, chips, n_jobs=1)

        markov = MarkovChain(k=1, use_prior=True, reset = False, prior=1., specific_prior=prior,
                                specific_prior_vocab = vocab, state_count=len(vocab), modus="bayes")
        markov.prepare_data(paths)
        markov.fit(paths)

        evidence = markov.bayesian_evidence()
        evidences[i] = evidence

    ax.plot(evidences.keys(), evidences.values(), marker='o', clip_on = False, label="self-loop", linestyle='-.')
    return evidences

def uniform_prior(paths, vocab, ax):

    # In this case we set each element of the hypothesis matrix to the same value
    # We only set the diagonal to zero as we do not consider self-loops
    shape = len(vocab)
    evidences = {}
    for i in xrange(5):
        if i == 0:
            prior = csr_matrix((shape,shape), dtype=np.float64)
        else:
            matrix_uniform = lil_matrix((shape,shape), dtype=np.float64)
            matrix_uniform[:] = 1.
            matrix_uniform.setdiag(0.)
            matrix_uniform = matrix_uniform.tocsr()

            chips = i*(matrix_uniform.shape[0]-1.)
            prior = distr_chips_row(matrix_uniform, chips, n_jobs=1)

        markov = MarkovChain(k=1, use_prior=True, reset = False, prior=1., specific_prior=prior,
                                specific_prior_vocab = vocab, state_count=len(vocab), modus="bayes")
        
        # If we consider self-loops, the whole process can be done more elegantly. In that case, all elicited
        # Dirichlet pseudo counts receive the same value. Thus, we can directly incorporate this into inference
        # via the prior parameter of the MarkovChain class as this is the pseudo count that each
        # single transition receives.
        # markov = pt.MarkovChain(k=1, use_prior=True, reset = False, prior=1.+i, specific_prior=prior,
        #                        specific_prior_vocab = vocab, modus="bayes")
        
        markov.prepare_data(paths)
        markov.fit(paths)
        evidence = markov.bayesian_evidence()
        evidences[i] = evidence

    ax.plot(evidences.keys(), evidences.values(), marker='o', clip_on = False, label="uniform", linestyle='-')
    return evidences

def data_prior(paths, vocab, ax):
    
    # Again we start by generating the hypothesis matrix G
    shape_matrix = (max(vocab.itervalues()) + 1, max(vocab.itervalues()) + 1)
    evidences = {}
    shape = len(vocab)
    for i in xrange(5):
        if i == 0:
            prior = csr_matrix((shape,shape), dtype=np.float64)
        else:
            # We first calculate the MLE for the data
            markov = MarkovChain(k=1, use_prior=False, reset = False, modus="mle")
            markov.prepare_data(paths)
            markov.fit(paths)
            mle = markov.transition_dict_
            
            # Now we use the parameter configuration from the MLE as a hypothesis
            i_indices = list()
            j_indices = list()
            values = list()
            for s,targets in mle.iteritems():
                for t, v in targets.iteritems():
                    i_indices.append(vocab[s[0]])
                    j_indices.append(vocab[t])
                    values.append(v)

            matrix_data = csr_matrix((values, (i_indices, j_indices)),
                             shape=shape_matrix, dtype=np.float64)
            
            # Typical chip distribution
            chips = i*(matrix_data.shape[0]-1.)
            prior = distr_chips_row(matrix_data, chips, n_jobs=1)

        markov = MarkovChain(k=1, use_prior=True, reset = False, prior=1., specific_prior=prior,
                                specific_prior_vocab = vocab, state_count=len(vocab), modus="bayes")
        markov.prepare_data(paths)
        markov.fit(paths)

        evidence = markov.bayesian_evidence()
        evidences[i] = evidence

    ax.plot(evidences.keys(), evidences.values(), marker='o', clip_on = False, label="data", linestyle='--')
    return evidences

def calc_latent(paths, X):
    
    states = set(range(X.shape[0]))
    vocab = dict(((t, i) for i, t in enumerate(states)))

# This is a simple pipeline for the calculation
def calc(paths, g):

    # First, we need to define the different states that we are interested in and
    # over which we build our hypotheses. In this case, these are simply all nodes
    # of the network. Note that this might also include states that cannot be observed
    # in the data.
    states = set()
    for n in g.nodes():
        states.add(n)
    
    # In this step we build a vocabulary which is important for the HypTrails framework
    # Concretely, we assign an index to each state. So for example, "Node 1": "0"
    # This should match the indices of the matrices that we work with to their corresponding states
    # Thus, this matching has to be kept in mind throughout all steps that follow
    vocab = dict(((t, i) for i, t in enumerate(states)))
    
    # Figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Now, let us calculate the evidences
    print "UNIFORM"
    print "-------"
    ret = uniform_prior(paths, vocab, ax)

    print "STRUCTURAL"
    print "-------"
    ret = struc_prior(paths, vocab, g, ax)
    
    #print "DATA"
    #print "-------"
    #ret = data_prior(paths, vocab, ax)
    
    #print "SELF-LOOP"
    #print "-------"
    #ret = selfloop_prior(paths, vocab, ax)

    # Further plotting
    ax.set_xlabel("hypothesis weighting factor k")
    ax.set_ylabel("marginal likelihood / evidence")

    plt.legend(loc=7, handlelength = 3)
    plt.grid(False)
    ax.xaxis.grid(True)
    plt.tight_layout(pad=0.2)
    plt.show()


n = 100
length = 5

g = nx.barabasi_albert_graph(10000, 10)
trails_s = create_structure_trails(g, n, length, 0, 1)
trails_u = create_uniform_trails(g, n, length, 5000, 10000)
trails = trails_s + trails_u
calc(trails, g)

fpath = 'iflux.tmp'
with open(fpath, 'w') as iflux_file:
    h = 'struct'
    for trail in trails_s:
        prev = None
        for o in trail:
            if prev is not None:
                d = o
                s = prev
                print >>iflux_file, '%d\t%s\t%d\t%d' % (1, h, s, d)
            prev = o

    #h = 'sloop'
    #for trail in trails_l:
    #    prev = None
    #    for o in trail:
    #        if prev is not None:
    #            d = o
    #            s = prev
    #            print >>iflux_file, '%d\t%s\t%d\t%d' % (1, h, s, d)
    #        prev = o
    
    h = 'uni'
    for trail in trails_u:
        prev = None
        for o in trail:
            if prev is not None:
                d = o
                s = prev
                print >>iflux_file, '%d\t%s\t%d\t%d' % (1, h, s, d)
            prev = o


from node_sherlock import learn
nz = 20
rv = learn.fit('iflux.tmp', nz, 50. / nz, 0.001, 0.001, 1, nz - 1, 800, 300)

Psi_sz = rv['Psi_sz']#.values
Psi_dz = rv['Psi_dz']#.values
count_z = rv['count_z']#.values[:, 0]
Psi_zs = (Psi_sz * count_z).T

dest2id = rv['dest2id']
source2id = rv['source2id']

id2dest = dict((v, k) for k, v in dest2id.items())
id2source = dict((v, k) for k, v in source2id.items())

for z in xrange(nz):
    psz = Psi_sz[:, z]
    pdz = Psi_dz[:, z]
    X = np.outer(pdz, psz)
    T = np.zeros(shape=(len(g.nodes()), len(g.nodes())), dtype='d')
    for row in xrange(X.shape[0]):
        for col in xrange(X.shape[1]):
            T[int(id2dest[row]), int(id2source[col])] = X[row, col]
    np.fill_diagonal(T, 0) 
    T = T / T.sum(axis=0)

    trails_z = []
    for i in xrange(n):
        path = []
        v = choice(g.nodes())
        path.append(v)
        for l in xrange(length - 1):
            p = T[:, v]
            v = p.argmax()
            path.append(v)
        
        trails_z.append(path)
    calc(trails_z, g)
