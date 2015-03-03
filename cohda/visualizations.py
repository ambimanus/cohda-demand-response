# coding=utf-8

from __future__ import division

import logger
from logger import *
import util


class Stats(object):
    def __init__(self, cfg, agents):
        self.cfg = cfg
        self.agents = agents
        self.solution = {}
        self.bkc_sizes = {}
        self.bkc_ratings = {}
        self.first_time = True
        self.new_solution = False
        self.distance = 1.0
        self.bkcmin = 1.0
        self.bkcmin_dist = 0.0
        self.bkcmin_size = 0.0
        self.sel = None
        self.eq = None
        self.bkc_sel = None
        self.bkc_history = {}
        self.full_bkc_ratings = {}
        self.switches = 0
        self.message_counter = 0
        self.agent_keys = sorted(self.agents.keys())
        self.aid_m = None
        self.active = False

        for aid in self.cfg.agent_ids:
            self.solution[aid] = self.cfg.sol_init[aid]


    def eval(self, current_time):
        d_min, d_max = self.cfg.sol_d_min, self.cfg.sol_d_max
        obj = self.cfg.objective

        self.active = False
        # Collect agent states
        for aid in self.agent_keys:
            a = self.agents[aid]
            self.active = self.active or a.dirty
            if self.solution[aid] != a.sol:
                self.solution[aid] = a.sol
                self.new_solution = True
            if a.bkc_f is not None:
                s = len(a.bkc)
                self.bkc_sizes[aid] = s
                s_n = s / self.cfg.opt_m
                if s_n > self.bkcmin_size:
                    self.bkcmin_size = s_n
                self.bkc_ratings[aid] = a.bkc_f
                if s_n == 1.0:
                    self.full_bkc_ratings[aid] = a.bkc_f
                else:
                    # Sanity check. This must not happen.
                    assert aid not in self.full_bkc_ratings
                bm_norm = util.norm(d_min, d_max, a.bkc_f)
                if s_n >= self.bkcmin_size and bm_norm < self.bkcmin:
                    self.bkcmin = bm_norm
                    self.aid_m = aid

        if self.new_solution:
            self.bkc_history[current_time] = self.bkcmin

        if logger.LOG_LEVEL >= 1000:
            return
        # sel := keys of bkc values that should be considered.
        # At the beginning of the simulation, report all bkc values.
        # But as soon as the first complete one has be found, restrict the
        # report to only complete bkc values.
        # In either case, consider only the set of the largest bkcs available.
        sel = []
        bm = max(self.bkc_sizes.values())
        for k in sorted(self.bkc_sizes.keys()):
            if self.bkc_sizes[k] == bm:
                sel.append(k)
        if len(self.bkc_ratings) > 0 and len(sel) > 0:
            bkc_sel = [self.bkc_ratings[k] for k in sel]
            # bkcmin := minimal found bkc value
            bkcmin = min(bkc_sel)
            # eq := keys of bkc_sel whith minimal value
            eq = []
            for k in sorted(self.bkc_ratings.keys()):
                if abs(self.bkc_ratings[k] - bkcmin) < 0.00001:
                    eq.append(k)
            bkcmin = util.norm(d_min, d_max, bkcmin)
            # bkcmin_dist := distribution of minimal bkc in population
            bkcmin_dist = len(eq) / self.cfg.opt_m
            # bkcmin_size := completeness of minimal bkc with respect to opt_m
            bkcmin_size = None
            for k in eq:
                s = self.bkc_sizes[k]
                if bkcmin_size is None or s > bkcmin_size:
                    bkcmin_size = s
            bkcmin_size = bkcmin_size / self.cfg.opt_m
            # Sanity check:
            if (bkcmin_size < self.bkcmin_size or
                    (bkcmin_size == self.bkcmin_size and
                     bkcmin > self.bkcmin)):
                ERROR('bkc convergence problem!')
                ERROR('previous values:')
                ERROR('  bkc_sizes:', self.bkc_sizes_bak)
                ERROR('  sel:', self.sel)
                ERROR('  bkc_sel:', self.bkc_sel)
                ERROR('  eq:', self.eq)
                ERROR('current values:')
                ERROR('  bkc_sizes:', self.bkc_sizes)
                ERROR('  sel:', sel)
                ERROR('  bkc_sel:', bkc_sel)
                ERROR('  eq:', eq)
            # Store values
            self.bkc_sel = bkc_sel
            self.bkcmin = bkcmin
            self.bkcmin_size = bkcmin_size
            self.bkcmin_dist = bkcmin_dist
            self.bkc_sizes_bak = dict(self.bkc_sizes)
            self.sel = sel
            self.eq = eq
            # Prevent rounding to 1.0 for display purposes
            if 0.99 < bkcmin_dist < 1.0:
                bkcmin_dist = 0.99
            if 0.99 < bkcmin_size < 1.0:
                bkcmin_size = 0.99

        # print runtime values
        if self.first_time:
            self.bkc_history[current_time] = self.bkcmin
            sol = sum(self.cfg.sol_init.values())
            INFO(' time |  distance | bkc-value | bkc-size | bkc-dist')
            SOLUTION('%5.1f' % 0.0,
                     '% .6f |' % util.norm(d_min, d_max,
                                           obj(sol, record_call=False)),
                     ' %.6f |' % self.bkcmin, '  %.2f   |' % self.bkcmin_size,
                     '  %.2f   |' % self.bkcmin_dist)
            self.first_time = False
        elif self.new_solution:
            self.bkc_history[current_time] = self.bkcmin
            sol = sum(self.solution.values())
            self.distance = util.norm(d_min, d_max,
                                      obj(sol, record_call=False))
            SOLUTION('%5.1f' % current_time, '% .6f |' % self.distance,
                     ' %.6f |' % self.bkcmin, '  %.2f   |' % self.bkcmin_size,
                     '  %.2f   |' % self.bkcmin_dist)
            self.new_solution = False
        else:
            STATS('%5.1f' % current_time, 'None      |',
                  ' %.6f |' % self.bkcmin, '  %.2f   |' % self.bkcmin_size,
                  '  %.2f   |' % self.bkcmin_dist)


    def is_active(self, current_time):
        # Check maximal runtime
        if (self.cfg.max_simulation_steps and
                current_time >= self.cfg.max_simulation_steps):
            INFO('Stopping (max simulation steps reached)')
            return False
        # Search final bkc
        # aid_m = None if len(self.full_bkc_ratings) == 0 else min(
        #         self.full_bkc_ratings.iterkeys(),
        #         key=(lambda key: self.bkc_ratings[key]))
        # Check mean bkc improvement over last x solutions
        # FIXME: This would stop the process in an unconverged state, so a
        #        post-processing step would be necessary to settle the current
        #        bkc in all agents.
        x = 10
        if (self.cfg.min_solution_gradient is not None and
                self.aid_m is not None and
                self.aid_m in self.full_bkc_ratings and
                self.bkc_history is not None and
                len(self.bkc_history) > x and
                current_time in self.bkc_history):
            grad = 0
            keys = sorted(self.bkc_history.keys())
            for i in range(x, 0, -1):
                s_pre = self.bkc_history[keys[-(i + 1)]]
                s_post = self.bkc_history[keys[-i]]
                grad += abs(s_pre - s_post)
            grad /= x
            if grad < self.cfg.min_solution_gradient and self.aid_m is not None:
                INFO('Stopping (min solution gradient reached)')
                self.solution = self.agents[self.aid_m].bkc
                return False
        # Check minimum solution distance criterion
        # FIXME: This would stop the process in an unconverged state, so a
        #        post-processing step would be necessary to settle the current
        #        bkc in all agents.
        if (self.cfg.min_solution_distance is not None and
                self.aid_m is not None and
                self.aid_m in self.full_bkc_ratings and
                self.bkcmin <= self.cfg.min_solution_distance):
            INFO('Stopping (min solution distance reached)')
            self.solution = self.agents[self.aid_m].bkc
            return False
        # Check agent activity
        if self.active:
            return True
        INFO('Stopping (no agent activity)')

        return False


    def is_converged(self):
        if self.bkcmin_dist != 1.0:
            return False
        for a in self.agents.values():
            if a.sol != a.bkc[a.aid]:
                return False
        return True


    def eval_final(self):
        # if not self.is_converged():
        #     ERROR('convergence not reached!')
        sol = sum(self.solution.values())
        for aid in sorted(self.agents.keys()):
            if self.cfg.sol_init[aid] != self.solution[aid]:
                self.switches += 1
        self.message_counter = logger.message_counter
        INFO('Target: %.2f' % self.cfg.objective.target)
        INFO('Result: %.2f' % sol)
        INFO('Abs. Distance: %.2f' % self.cfg.objective(sol, record_call=False))
        INFO('Number of switched states: %d out of %d (%d%%)' % (
                self.switches, len(self.agents),
                self.switches * 100 / len(self.agents)))
