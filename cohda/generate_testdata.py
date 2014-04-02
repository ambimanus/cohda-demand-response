# coding=utf-8

from __future__ import division

import sys, os, random


P = [0, 3000]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python %s <seed> <number of units>' % os.path.basename(sys.argv[0])
        sys.exit(1)

    seed = int(sys.argv[1])
    opt_m = int(sys.argv[2])

    print '%d,-1,-1,-1,-1' % seed
    # 60% of the aggregated rated power
    print '%.2f,-1,-1,-1,-1' % (opt_m * (P[0] + 0.66 * (P[1] - P[0])))

    rnd = random.Random(seed)
    for i in range(opt_m):
        p_refuse = rnd.random()
        sample = sorted(rnd.sample(P, rnd.randint(1, 2)))
        if len(sample) == 1:
            sample.append(sample[-1])
        # ID,STATE,<LIST OF FEASIBLE VALUES>
        print ','.join(map(str, [i, rnd.choice(P), p_refuse] + sample))
