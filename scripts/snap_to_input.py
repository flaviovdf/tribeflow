#-*- coding: utf8
'''
Converts a temporal graph from the snap dataset to our input
'''
from __future__ import division, print_function

import gzip
import plac
import os

def main(edges_fpath, date_fpath):
    
    years = {}
    with gzip.open(date_fpath) as dates_file:
        for line in dates_file:
            spl = line.split()
            node_id = spl[0]
            node_year = spl[1].split('-')[0]
            years[node_id] = node_year
    
    with gzip.open(edges_fpath) as edges_file:
        for line in edges_file:
            if '#' == line[0]:
                continue
            
            from_, to = line.split()
            if from_ in years:
                print(years[from_], from_, to, 1, sep='\t')

if __name__ == '__main__':
    plac.call(main)
