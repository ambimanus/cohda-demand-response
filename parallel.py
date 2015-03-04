# coding=utf-8

from __future__ import division

import sys
import os
import pickle

from configuration import Configuration
import main

def run(cfg_dict):
    cfg = main.main(Configuration(**cfg_dict))
    fn = str(os.path.join(cfg.basepath, '.'.join(
            ('cfg', cfg.title, str(cfg.seed), 'pickle'))))
    with open(fn, 'w') as f:
        pickle.dump(cfg, f)


if __name__ == '__main__':
    runs = 10
    params = []

    for lag in range(0, 6):
        for seed in range(runs):
            cfg_dict = {
                'title': 'lag%02d' % lag,
                'seed': seed,
                'n': 1500,
                'it': 1441,
                'lag': lag,
            }
            params.append(cfg_dict)


    if 'SGE_TASK_ID' in os.environ:
        # HERO HPC cluster
        run(params[int(os.environ['SGE_TASK_ID'])-1])
    elif 'PARALLEL_SEQ' in os.environ:
        # GNU parallel
        run(params[int(os.environ['PARALLEL_SEQ'])-1])
    else:
        # sequential
        start, stop = 0, len(params)
        if len(sys.argv) == 2:
            print len(params)
            sys.exit(0)
        elif len(sys.argv) == 3:
            start, stop = int(sys.argv[1]), int(sys.argv[2])
        if start >= len(params):
            sys.exit(0)
        for p in params[start:stop]:
            run(p)
