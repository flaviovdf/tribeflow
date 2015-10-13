#-*- coding: utf8
from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.basemap import Basemap

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#API_KEY = 'AIzaSyBv2cM4hW0NU15-LFCgJe-0ILHj8N7_nQ0'
#from googleplaces import GooglePlaces, types, lang
#google_places = GooglePlaces(API_KEY)

def plot_map(lat, lon, weight):
    lat = np.asanyarray(lat)
    lon = np.asanyarray(lon)
    weight = np.asanyarray(weight)
    
    idx = (lat > -90) & (lat < 90) & (lon > -180) & (lon < 180)
    outl = weight.mean() + 2 * weight.std()

    idx = idx & (weight > outl)
    lat = lat[idx]
    lon = lon[idx]
    weight = weight[idx]

    #my_map = Basemap(projection='robin', resolution='l', area_thresh=1000.0, \
    #        lat_0=np.median(lat), lon_0=np.median(lon), \
    #        llcrnrlon=lon.min(), llcrnrlat=lat.min(), \
    #        urcrnrlon=lon.max(), urcrnrlat=lat.max())
    
    my_map = Basemap(projection='robin', resolution='l', area_thresh=1000.0, \
            lat_0=0, lon_0=0, \
            llcrnrlon=-180, llcrnrlat=-90, \
            urcrnrlon=180, urcrnrlat=90)
 
    my_map.drawcoastlines(linewidth=0.6)
    my_map.drawcountries()
    #my_map.fillcontinents(color='#F8F8F8')
    #my_map.drawmapboundary(fill_color='#F0FFFF')
    my_map.fillcontinents(color='#FFFFFF')
    my_map.drawmapboundary(fill_color='#FFFFFF')
    
    for i in weight.argsort()[::-1][:10]:
        #query_result = google_places.nearby_search(\
        #        lat_lng={'lat':lat[i], 'lng':lon[i]}, \
        #        radius=3200, types=[types.])
        #print(query_result.places)
        print(lat[i], lon[i])#, query_result.places[0].name)

    x, y = my_map(lon, lat)
    my_map.plot(x, y, 'ro', markersize=6)
    #cb = my_map.hexbin(x, y, C=weight, reduce_C_function=np.mean, gridsize=200, \
    #        bins='log', cmap=plt.cm.jet)
    #cb.set_label('log probability')
    #my_map.colorbar()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_fpath', help='The name of the model file (a h5 file)', \
            type=str)
    parser.add_argument('lat_long_fpath', help='Place id to lat long', \
            type=str)
    #parser.add_argument('user_lat_long_fpath', help='User id to lat long', \
    #        type=str)
    args = parser.parse_args()
    model = pd.HDFStore(args.model_fpath, 'r')         
    
    assign = model['assign'].values[:, 0]
    deltas = model['tstamps'].values[:, 0]
    
    Theta_zh = model['Theta_zh'].values
    Theta_hz = Theta_zh.T * model['count_z'].values[:, 0]
    Theta_hz = Theta_hz / Theta_hz.sum(axis=0)
    
    Psi_oz = model['Psi_sz'].values

    hyper2id = model['hyper2id'].values
    obj2id = model['source2id'].values
    
    id2user = dict((r[1], r[0]) for r in hyper2id)
    id2obj = dict((r[1], r[0]) for r in obj2id)
    
    #user2latlong = {}
    #with open(args.user_lat_long_fpath) as lat_long_file:
    #    for line in lat_long_file:
    #        user, lat, long_ = line.split('\t')
    #        user2latlong[user] = (float(lat), float(long_))

    loc2latlong = {}
    with open(args.lat_long_fpath) as lat_long_file:
        for line in lat_long_file:
            loc, lat, long_ = line.split('\t')
            loc2latlong[loc] = (float(lat), float(long_))

    for z in xrange(Theta_zh.shape[0]):
        print(z)

        #plt.subplot(211)
        #lats = []
        #lons = []
        #weights = []
        #print('Users')
        #for x in xrange(Theta_zh.shape[1]):
        #    if id2user[x] in user2latlong:
        #        lats.append(user2latlong[id2user[x]][0])
        #        lons.append(user2latlong[id2user[x]][1])
        #        weights.append(Theta_hz[x, z])
        #plot_map(lats, lons, weights)
        #plt.title('Top Users')
        #print()

        #plt.subplot(212)
        lats = []
        lons = []
        weights = []
        #print('Source')
        for x in xrange(Psi_oz.shape[0]):
            if id2obj[x] in loc2latlong:
                lats.append(loc2latlong[id2obj[x]][0])
                lons.append(loc2latlong[id2obj[x]][1])
                weights.append(Psi_oz[x, z])
        plot_map(lats, lons, weights)
        #plt.title('Top Sources')
        print()
        
        plt.tight_layout(pad=0)
        print('z_%d.pdf' % z)
        plt.savefig('z_%d.pdf' % z)
        plt.close()
        print('\n\n')

    model.close()

if __name__ == '__main__':
    main()
