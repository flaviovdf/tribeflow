from __future__ import print_function

from collections import OrderedDict
import sys

LEAVE_OUT = 0.3
with open(sys.argv[1]) as in_file:
    num_lines = sum(1 for _ in in_file)
to = int(num_lines * (1 - LEAVE_OUT))

obj2id = {}

u_trans = OrderedDict()
u_trans_test = OrderedDict()

with open(sys.argv[1]) as in_file:
    for i, line in enumerate(in_file):
        train = i < to
        _, u, s, d = line.split()
        
        if train:
            if s not in obj2id:
                obj2id[s] = len(obj2id)
            
            if d not in obj2id:
                obj2id[d] = len(obj2id)

            if u not in u_trans:
                u_trans[u] = []
            
            if len(u_trans[u]) == 0:
                u_trans[u].append(obj2id[s])
                u_trans[u].append(obj2id[d])
            else:
                u_trans[u].append(obj2id[d])
        else:
            if s in obj2id and d in obj2id:
                if obj2id[s] not in counts_test:
                    counts_test[obj2id[s]] = 0
                
                if obj2id[d] not in counts_test:
                    counts_test[obj2id[d]] = 0

                if u not in u_trans_test:
                    u_trans_test[u] = []

                if len(u_trans_test[u]) == 0:
                    u_trans_test[u].append(obj2id[s])
                    u_trans_test[u].append(obj2id[d])
                else:
                    u_trans_test[u].append(obj2id[d])


for u in u_trans:
    print(u, end=' ')
    print(' '.join(str(x) for x in u_trans[u]), end='\n')

for u in u_trans_test:
    print(u, end=' ')
    print(' '.join(str(x) for x in u_trans_test[u]), end='\n', file=sys.stderr)
