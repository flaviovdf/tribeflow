#!-*- coding: utf8

from collections import OrderedDict

from numpy.linalg import lstsq

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import os
import pandas as pd
import plac

def main(trace_fpath, leaveout=0.3):
    leaveout = float(leaveout)
    df = pd.read_csv(trace_fpath, sep='\t', names=['dt', 'u', 's', 'd'])
    
    num_lines = len(df)
    to = int(num_lines - num_lines * leaveout)
    
    df_train = df[:to]
    df_test = df[to:]
    
    documents_train_right = OrderedDict()
    documents_train_left = OrderedDict()
    tokens_train = set()
    for _, u, s, d in df_train.values:
        u = str(u)
        s = str(s)
        d = str(d)
        if u not in documents_train_right:
            documents_train_right[u] = []
            documents_train_left[u] = []

        documents_train_right[u].append(s)
        documents_train_left[u].append(d)
        tokens_train.add(s)
    
    for u in documents_train_right:
        documents_train_right[u] = '\t'.join(documents_train_right[u])
        documents_train_left[u] = '\t'.join(documents_train_left[u])

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split('\t'), \
            vocabulary=tokens_train)
    X_train_counts = vectorizer.fit_transform(documents_train_right.values())
    Y_train_counts = vectorizer.transform(documents_train_left.values())
    
    lda_model = LatentDirichletAllocation(n_topics=100, n_jobs=-1)
    lda_model.fit(X_train_counts)
    
    Theta_zh = lda_model.transform(X_train_counts).T
    ph = X_train_counts.sum(axis=1)
    pz = np.asarray(Theta_zh.dot(ph))[:, 0]

    Psi_oz = lda_model.components_.T
    pz = pz / pz.sum()
    Psi_zo = (Psi_oz * pz).T

    #Normalize matrices
    Psi_oz = Psi_oz / Psi_oz.sum(axis=0)
    Psi_zo = Psi_zo / Psi_zo.sum(axis=0)
    
    X_train_probs = []
    Y_train_probs = []
    for _, u, s, d in df_train.values:
        if str(s) in vectorizer.vocabulary_ and str(d) in vectorizer.vocabulary_:
            id_s = vectorizer.vocabulary_.get(str(s))
            id_d = vectorizer.vocabulary_.get(str(d))
            X_train_probs.append(Psi_zo[:, id_s])
            Y_train_probs.append(Psi_zo[:, id_d])
    
    X_train_probs = np.array(X_train_probs)
    Y_train_probs = np.array(Y_train_probs)    
    P_zz = lstsq(X_train_probs, Y_train_probs)[0].T
    
    #numerical errors, expected as in paper.
    P_zz[P_zz < 0] = 0
    
    I = Psi_oz.dot(P_zz) 
    I = I / I.sum(axis=0)

    probs_tmlda = {}
    probs_lda = {}

    ll_tmlda = 0.0
    ll_lda = 0.0
    n = 0
    for _, u, s, d in df_test.values:
        u = str(u)
        s = str(s)
        d = str(d)
        if s in vectorizer.vocabulary_ and d in vectorizer.vocabulary_:
            id_s = vectorizer.vocabulary_.get(s)
            id_d = vectorizer.vocabulary_.get(d)
            if (id_d, id_s) not in probs_tmlda:
                probs_tmlda[id_d, id_s] = (Psi_zo[:, id_s] * I[id_s]).sum() 
                probs_lda[id_d, id_s] = (Psi_zo[:, id_s] * Psi_oz[id_s]).sum() 
            
            if probs_tmlda[id_d, id_s] != 0:
                ll_tmlda += np.log(probs_tmlda[id_d, id_s])
            if probs_lda[id_d, id_s] != 0:
                ll_lda += np.log(probs_lda[id_d, id_s])
            n += 1

    print(ll_tmlda, ll_lda)
    print(ll_tmlda / n, ll_lda / n)

if __name__ == '__main__':
    plac.call(main)
