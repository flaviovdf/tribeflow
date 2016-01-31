#-*- coding: utf8
from __future__ import division, print_function

from collections import defaultdict

import plac

def read_results(results_fpath, lamb):
    
    last = {}
    classes = {}
    objs = set()
    users = set()
    
    joint = defaultdict(lambda: defaultdict(int))
    glob = defaultdict(lambda: defaultdict(int))
    
    with open(results_fpath) as results_file:
        results_file.readline()
        for line in results_file:
            spl = line.strip().split()
            user = spl[0]
            class_ = spl[1]

            last_stage = spl[-1].split('-')[-1]
            last[user] = last_stage
            classes[user] = class_

            for item in spl[2:]:
                obj, stage = item.split('-')
                joint[class_][stage, obj] += 1
                glob[class_][stage] += 1
                objs.add(obj)
            users.add(user)

    probs = {}
    for u in last:
        probs[u] = {}
        class_ = classes[u]
        for o in objs:
            probs[u][o] = (lamb + joint[class_][last[u], o]) / \
                    (len(objs) * lamb + glob[class_][last[u]])

    for u in probs:
        sum_ = sum(probs[u].values())
        for o in objs:
            probs[u][o] = probs[u][o] / sum_ if sum_ > 0 else 0

    sorted_probs = {}
    for u in probs:
        sorted_probs[u] = sorted([(v, o) for o, v in probs[u].items()],
                reverse=True)
    return sorted_probs, users, objs

def main(results_fpath, test_fpath):
    lamb = 1.0
    probs, users, objs = read_results(results_fpath, lamb)
    rrs = []
    with open(test_fpath) as test_file:
        for line in test_file:
            spl = line.strip().split()
            user = spl[0]
            if user not in users:
                continue

            for o in spl[1:]:
                rr = 1.0
                for v, c in probs[user]:
                    if c == o:
                        break
                    else:
                        rr += 1
                rr = 1.0 / rr
                rrs.append(rr)
    print(sum(rrs) / len(rrs))

if __name__ == '__main__':
    plac.call(main)
