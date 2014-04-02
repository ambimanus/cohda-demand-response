# coding=utf-8

import logging

from logger import *
from util import PBar


class Simulator(object):
    def __init__(self, cfg, agents):
        self.cfg = cfg
        self.agents = agents
        self.current_time = 0


    def init(self):
        speaker = (self.cfg.rnd.choice(self.agents.keys())
                   if self.cfg.random_speaker else 0)
        INFO('Notifying speaker (a%d)' % speaker)
        objective = self.cfg.objective
        self.agents[speaker].notify(objective)
        objective._reset_call_counter()


    def step(self):
        if LOG_LEVEL <= logging.INFO:
            progress = PBar(self.cfg.opt_m).start()
        for a in self.agents.values():
            a.step()
            if LOG_LEVEL <= logging.INFO:
                progress.update()
        self.current_time += 1
