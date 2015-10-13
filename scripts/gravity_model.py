#!-*- coding: utf8
from scipy.stats import linregress

import matplotlib
matplotlib.use('Agg')

from matplotlib import rc

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import plac
import statsmodels.api as sm

C = math.pi / 180.0

def initialize_matplotlib():

    inches_per_pt = 1.0 / 72.27
    fig_width = 120 * inches_per_pt  # width in inches
    fig_height = 96 * inches_per_pt  #.4 * fig_width
    
    rc('axes', labelsize=6)
    rc('axes', titlesize=6)
    rc('axes', unicode_minus=False)
    rc('axes', grid=False)
    rc('figure', figsize=(fig_width, fig_height))
    rc('grid', linestyle=':')
    rc('font', family='serif')
    rc('legend', fontsize=5)
    rc('lines', linewidth=.7)
    rc('ps', usedistiller='xpdf')
    rc('text', usetex=True)
    rc('xtick', labelsize=6)
    rc('ytick', labelsize=6)

def distance(s, d, lat_long_dict):

    lat1, long1 = lat_long_dict[s]
    lat2, long2 = lat_long_dict[d]
    
    if (lat1, long1) == (lat2, long2):
        return 0.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * C 
    phi2 = (90.0 - lat2) * C 
         
    # theta = longitude
    theta1 = long1 * C
    theta2 = long2 * C
         
    # Compute spherical distance from spherical coordinates.
         
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
     
    cos = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) + \
            math.cos(phi1) * math.cos(phi2))
    arc = math.acos(cos)
 
    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return arc

def main(trace_fpath, lat_long_fpath, leaveout=0.3):
    initialize_matplotlib()
    leaveout = float(leaveout)
    
    lat_long_dict = {}
    with open(lat_long_fpath) as lat_long_file:
        for line in lat_long_file:
            loc, lat, long_ = line.split('\t')
            lat_long_dict[loc] = (float(lat), float(long_))

    df = pd.read_csv(trace_fpath, sep='\t', names=['dt', 'u', 's', 'd'])
    
    num_lines = len(df)
    to = int(num_lines - num_lines * leaveout)
    
    df_train = df[:to]
    df_test = df[to:]
    
    pop_df = df_train.groupby(['d']).count()['u']
    pop_dict = dict(zip(pop_df.index, pop_df.values))
    answer_df = df_train.groupby(['s', 'd']).count()['u']
    answer_dict = dict(zip(answer_df.index, answer_df.values))
    
    X = []
    y = []
    for row in df_train[['s', 'd']].values:
        s, d = row
        if s in pop_dict and d in pop_dict and \
                str(s) in lat_long_dict and str(d) in lat_long_dict:
            dist = distance(str(s), str(d), lat_long_dict)
            if dist == 0: #different ids, same loc, ignore
                continue

            X.append([1.0, np.log(pop_dict[s]), np.log(pop_dict[d]), -np.log(dist)])
            y.append(answer_dict[s, d])
    
    answer_df_test = df_test.groupby(['s', 'd']).count()['u']
    answer_dict_test = dict(zip(answer_df_test.index, answer_df_test.values))
    
    #This is future information, should not be exploited for likelihood
    pop_df_test = df_test.groupby(['d']).count()['u']
    pop_dict_test = dict(zip(pop_df_test.index, pop_df_test.values))

    X_test_ll = []
    X_test_pred = []

    y_test = []
    for row in df_test[['s', 'd']].values:
        s, d = row
        if s in pop_dict and d in pop_dict and \
                str(s) in lat_long_dict and str(d) in lat_long_dict:

            dist = distance(str(s), str(d), lat_long_dict)
            if dist == 0: #different ids, same loc, ignore
                continue
            
            X_test_ll.append([1.0, np.log(pop_dict[s]), np.log(pop_dict[d]), \
                    -np.log(dist)])
            X_test_pred.append([1.0, np.log(pop_dict_test[s] if s in pop_dict_test else 0), \
                    np.log(pop_dict_test[d]), -np.log(dist)])
            y_test.append(answer_dict_test[s, d])

    X_train = np.asarray(X)
    y_train = np.asarray(y)

    X_test_ll = np.asarray(X_test_ll)
    X_test_pred = np.asarray(X_test_pred)

    y_test = np.asarray(y_test)
    
    import time
    print('training', time.localtime())
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
    results = model.fit()
    print('done', time.localtime())
    print(results.summary()) 
    
    y_pred = np.array(results.predict(X_test_pred))
    print(np.abs(y_test - y_pred).mean())
    
    plt.plot(y_pred, y_test, 'wo', rasterized=True, markersize=2)
    plt.plot(y_pred, y_pred, 'r-', rasterized=True)
    plt.minorticks_off()

    ax = plt.gca()
    ax.tick_params(direction='out', pad=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.ylabel(r'True value ($n_{ds}$)', labelpad=0.2)
    plt.xlabel(r'Predicted value ($\tilde{n_{ds}}$)', labelpad=0.3)
    plt.tight_layout(pad=0.1)
    _, _, r, _, _ = linregress(y_pred, y_test)
    plt.title('MAE = %.3f ; R2 = %.3f ' %(np.abs(y_test - y_pred).mean(), r**2), y=0.8)
    plt.savefig('pred.pdf')

    #Likelihood on test set (adapted from glm code on train set,
    #no method for test set exists)
    lin_pred = np.dot(X_test_ll, results.params) + model._offset_exposure
    expval = model.family.link.inverse(lin_pred)
    llr = model.family.loglike(expval, y_test, results.scale)
    llr = llr
    print(llr, llr / X_test_ll.shape[0]) 

if __name__ == '__main__':
    plac.call(main)
