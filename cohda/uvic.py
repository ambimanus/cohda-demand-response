# coding=utf-8

import sys
import os
import datetime
import random
import csv
import logging

import util
from configuration import Configuration
from logger import *
from agent import Agent
from visualizations import Stats
from simulator import Simulator


if __name__ == '__main__':
    sc_file = sys.argv[1]
    basedir = os.path.dirname(sc_file)
    setup_logger()
    ts = datetime.datetime.now().replace(microsecond=0).isoformat('_')
    INFO('Init %s' % ts)

    INFO('Importing %s' % sc_file)
    rnd, seed, target, sol_init, p_refuse, opt_w = None, None, None, {}, {}, {}
    with open(sc_file, 'rb') as scf:
        reader = csv.reader(scf, delimiter=',')
        for row in reader:
            if seed is None:
                seed = int(row[0])
                rnd = random.Random(seed)
            elif target is None:
                target = float(row[0])
            else:
                uid = int(row[0])
                state = float(row[1])
                p_refuse[uid] = float(row[2])
                opt_w[uid] = map(float, row[3:])
                # try to stay in previous state, if feasible
                if state in opt_w[uid]:
                    sol_init[uid] = state
                else:
                    sol_init[uid] = random.choice(opt_w[uid])

    cfg = Configuration(target, sol_init, opt_w, rnd, seed=seed)
    Objective.calls = 0
    Objective.instance = 0

    def pr(name, low, high, val):
        return '%s=%f [normalized %f]' % (name, val, util.norm(low, high, val))

    INFO(pr('d_min', cfg.sol_d_min, cfg.sol_d_max, cfg.sol_d_min))
    INFO(pr('d_max', cfg.sol_d_min, cfg.sol_d_max, cfg.sol_d_max))

    agents = dict()
    INFO('Creating %d agents' % cfg.opt_m)
    for i in range(cfg.opt_m):
        uid = cfg.agent_ids[i]
        # Start agent
        a = Agent(uid, opt_w[uid], cfg.sol_init[uid],
            cfg.rnd.randint(cfg.agent_delay_min, cfg.agent_delay_max),
            cfg.rnd.randint(0, cfg.max_seed), p_refuse[uid])
        agents[uid] = a
    # connect agents
    INFO('Connecting agents')
    for a, neighbors in cfg.network.items():
        for n in neighbors:
            DEBUG('', 'Connecting', n, '->', a)
            agents[a].add_peer(n, agents[n])

    sim = Simulator(cfg, agents)

    INFO('Starting simulation')
    stats = Stats(cfg, agents)
    sim.init()
    stats.eval(sim.current_time)
    while stats.is_active(sim.current_time):
        sim.step()
        stats.eval(sim.current_time)
    stats.eval_final()

    if len(sys.argv) > 2 and sys.argv[2] == '--noout':
        dfn = None
    else:
        dfn = '%s_result.csv' % sc_file[:-4]
        if os.path.exists(dfn):
            WARNING('File already exists: %s' % dfn)
        with open(dfn, 'wb') as df:
            writer = csv.writer(df, delimiter=',')
            for uid in cfg.agent_ids:
                r = stats.solution[uid]
                writer.writerow([uid, r])
        INFO('Result stored in %s' % dfn)

    ts = datetime.datetime.now().replace(microsecond=0).isoformat('_')
    INFO('End %s' % ts)

    # if LOG_LEVEL > logging.INFO and dfn is not None:
    #     print dfn
