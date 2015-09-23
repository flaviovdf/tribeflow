from __future__ import print_function

from collections import OrderedDict
import sys

LEAVE_OUT = 0.3
with open(sys.argv[1]) as in_file:
    num_lines = sum(1 for _ in in_file)
to = int(num_lines * (1 - LEAVE_OUT))

obj2id = {}

counts = {}
u_trans = OrderedDict()

counts_test = {}
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
            
            if obj2id[s] not in counts:
                counts[obj2id[s]] = 0
            
            if obj2id[d] not in counts:
                counts[obj2id[d]] = 0

            counts[obj2id[s]] += 1
            counts[obj2id[d]] += 1
            
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

                counts_test[obj2id[s]] += 1
                counts_test[obj2id[d]] += 1
                
                if u not in u_trans_test:
                    u_trans_test[u] = []

                if len(u_trans_test[u]) == 0:
                    u_trans_test[u].append(obj2id[s])
                    u_trans_test[u].append(obj2id[d])
                else:
                    u_trans_test[u].append(obj2id[d])


print(' '.join(str(x) for x in range(len(obj2id))))
print(' '.join(str(counts[x]) for x in range(len(obj2id))))
for u in u_trans:
    print(' '.join(str(x) for x in u_trans[u]), end=' \n')

print(' '.join(str(x) for x in range(len(obj2id))), file=sys.stderr)
print(' '.join(str(counts[x]) for x in range(len(obj2id))), file=sys.stderr)
for u in u_trans_test:
    print(' '.join(str(x) for x in u_trans_test[u]), end=' \n', file=sys.stderr)
