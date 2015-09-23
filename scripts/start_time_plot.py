#-*- coding: utf8
from __future__ import division, print_function

from collections import defaultdict

import argparse
import matplotlib.pyplot as plt
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_trace', \
            help='The name of the original trace', type=str)
    parser.add_argument('tstamp_column', \
            help='The column of the time stamp', type=int)
    parser.add_argument('hypernode_column', \
            help='The column of the time hypernode', type=int)
    parser.add_argument('obj_node_column', 
            help='The column of the object node', type=int)
    parser.add_argument('-d', '--delimiter', help='The delimiter', \
            type=str, default=None)
    parser.add_argument('-l', '--loops', help='Consider loops', \
            type=bool, default=False)
    parser.add_argument('-r', '--sort', help='Sort the trace', \
            type=bool, default=True)
    parser.add_argument('-f', '--fmt', \
            help='The format of the date in the trace', type=str, default=None)
    parser.add_argument('-s', '--scale', \
            help='Scale the time by this value', type=float, default=1.0)
    parser.add_argument('-k', '--skip_header', \
            help='Skip these first k lines', type=int, default=0)
    args = parser.parse_args()
    
    delim = args.delimiter
    hcol = args.hypernode_column
    ocol = args.obj_node_column
    tcol = args.tstamp_column
    fmt = args.fmt
    consider_loops = args.loops
    skip = args.skip_header

    def parser(s):
        if fmt:
            return time.mktime(time.strptime(s, fmt))
        else:
            return float(s)
    
    def gen():
        with open(args.original_trace) as trace_file:
            for _ in xrange(skip):
                trace_file.readline()

            for line in trace_file:
                spl = line.split(delim)
                h = spl[hcol].strip()
                o = spl[ocol].strip()
                t = spl[tcol].strip()
                
                if t.strip() and h.strip() and o.strip():
                    t = parser(t.strip())
                    yield [t, h, o]
             
    last = {}
    if args.sort:
        trace = lambda: sorted(gen())
    else:
        trace = lambda: gen()

    pop = defaultdict(int)
    seen = defaultdict(set)
    user_times = defaultdict(list)

    mem_size = 1
    first_tick = None
    for t_now, h_now, o_now in trace():
        if not first_tick:
            first_tick = t_now
        
        if h_now in last and len(last[h_now]) == mem_size:
            objs = (o_now, ) + tuple(last[h_now])
            pop[objs] += 1
            if h_now not in seen[objs]:
                seen[objs].add(h_now)
                user_times[objs].append((h_now, (t_now - first_tick) / (60 * 60 * 24)))
        else:
            last[h_now] = []

        last[h_now].append(o_now)
        if len(last[h_now]) > mem_size:
            del last[h_now][0]

    most_popular = sorted((v, k) for k, v in pop.items())[-3000:]
    styles = ['wo', 'rs', 'b^', 'm*', 'yD']
    i = 0
    x = []
    y = []
    for _, objs in most_popular:
        ticks = user_times[objs]
        x.append([])
        y.append([]) 
        for j in xrange(len(ticks)):
            y[i].append(i + 1)
            x[i].append(ticks[j][1])
        i += 1
    
    for i in xrange(len(most_popular)):
        plt.hexbin(x[i], y[i], cmap=plt.cm.hot_r, gridsize=50, \
                bins='log')
    
    plt.ylabel('Transition ID')
    plt.xlabel('Dataset age (in days)')
    plt.savefig('test.png')

if __name__ == '__main__':
    main()
