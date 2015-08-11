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
                h = spl[hcol]
                o = spl[ocol]
                t = spl[tcol]
                
                if t.strip() and h.strip() and o.strip():
                    t = parser(t.strip())
                    yield [t, h, o]
             
    last = {}
    if args.sort:
        trace = sorted(gen())
    else:
        trace = gen()

    for t_now, h_now, o_now in trace:
        if h_now in last:
            t_prev, o_prev = last[h_now]
            if o_prev == o_now and not consider_loops:
                continue
            print(t_now - t_prev, h_now.strip(), o_now.strip(), \
                    sep='\t')
        last[h_now] = (t_now, o_now)

if __name__ == '__main__':
    main()
