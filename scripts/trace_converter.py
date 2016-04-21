#-*- coding: utf8
from __future__ import division, print_function

import argparse
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
    parser.add_argument('-m', '--mem_size', \
            help='Memory Size (the markov order is m - 1)', type=int, default=1)
    args = parser.parse_args()
    
    delim = args.delimiter
    hcol = args.hypernode_column
    ocol = args.obj_node_column
    tcol = args.tstamp_column
    fmt = args.fmt
    consider_loops = args.loops
    skip = args.skip_header
    mem_size = args.mem_size

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
                h = spl[hcol]
                o = spl[ocol]
                t = spl[tcol]
                
                if t.strip() and h.strip() and o.strip():
                    t = parser(t.strip())
                    yield [t, h, o]
             
    last = {}
    if bool(args.sort):
        trace = sorted(gen())
    else:
        trace = gen()
    
    for t_now, h_now, o_now in trace:
        if h_now in last and len(last[h_now]) == mem_size:
            t_prev, o_prev = last[h_now][-1]
            if o_prev == o_now and not consider_loops:
                continue
            
            mem = last[h_now]
            for i in xrange(1, mem_size):
                print(mem[i][0] - mem[i - 1][0], end='\t')
            
            #print(t_now - t_prev, end='\t')
            print(t_now, end='\t')
            print(h_now.strip(), end='\t')
            for i in xrange(mem_size):
                print(mem[i][1].strip(), end='\t')
            
            print(o_now.strip())
        elif h_now not in last:
            last[h_now] = []
        
        append = len(last[h_now]) == 0 or \
                consider_loops or \
                ((not consider_loops) and o_now != last[h_now][-1][1]) 
        
        if append:
            last[h_now].append((t_now, o_now))
            if len(last[h_now]) > mem_size:
                del last[h_now][0]

if __name__ == '__main__':
    main()
