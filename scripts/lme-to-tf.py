import sys
with open(sys.argv[1]) as trace_file:
    trace_file.readline()
    trace_file.readline()
    for h, l in enumerate(trace_file):
        objs = [int(x) for x in l.strip().split()]
        for pair in zip(objs[:-1], objs[1:]):
            if pair[0] != pair[1]:
                print '%d\t%d\t%d\t%d' % (1, h, pair[0], pair[1])
