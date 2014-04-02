# coding=utf-8

from __future__ import division

from util import autoassign
import objectives
import networks
import util


class Configuration(object):

    @autoassign
    def __init__(self,
                 target,
                 sol_init,
                 opt_w,
                 rnd,
                 seed=0,
                 max_seed=2**16,
                 max_simulation_steps=None,
                 min_solution_gradient=0.001,
                 agent_delay_min=1,
                 agent_delay_max=2,
                 random_speaker=True,
                 network_type='smallworld',
                 network_c=3,
                 network_k=1,
                 network_phi=0.5):
        self.opt_m = len(opt_w)
        self.objective = objectives.Objective(target)

        self.sol_d_max, self.sol_d_min = util.bounds(
                self.opt_w, self.opt_m, self.objective, zerobound=True)

        self.agent_ids = sorted(opt_w.keys())
        self.network = getattr(networks, network_type)(self)
