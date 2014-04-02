# coding=utf-8

class Objective(object):

    instance = 0
    calls = 0

    def __init__(self, target):
        Objective.instance += 1
        self.instance = Objective.instance
        self.target = target

    def __call__(self, x, record_call=True):
        if record_call:
            Objective.calls += 1
        return abs(self.target - x)

    def __repr__(self):
        return str({'instance': self.instance, 'target': self.target})

    def _reset_call_counter(self):
        Objective.calls = 0
