#-*- coding: utf8
from __future__ import division, print_function

import plac

def main(trace_fpath, amount=1):
    amount = int(amount)
    
    last_lines = {}
    for line in open(trace_fpath):
        dt, user, source, dest = line.strip().split('\t')
        
        if user not in last_lines:
            last_lines[user] = []
        else:
            assert source == last_lines[user][-1][-1]

        last_lines[user].append((dt, source, dest))
        if len(last_lines[user]) == amount + 1:
            print('\t'.join(mem[0] for mem in last_lines[user]), end='\t')
            print(user, end='\t')
            print('\t'.join(mem[1] for mem in last_lines[user])[:-1], end='\t')
            print(dest)
            del last_lines[user][0]

if __name__ == '__main__':
    plac.call(main)
