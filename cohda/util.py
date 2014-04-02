# coding=utf-8

from __future__ import division

import inspect
from functools import wraps
from itertools import izip, ifilter, starmap
import time

import progressbar.progressbar as pbar


# http://code.activestate.com/recipes/551763/
def autoassign(*names, **kwargs):
    """
    autoassign(function) -> method
    autoassign(*argnames) -> decorator
    autoassign(exclude=argnames) -> decorator

    allow a method to assign (some of) its arguments as attributes of
    'self' automatically.  E.g.

    >>> class Foo(object):
    ...     @autoassign
    ...     def __init__(self, foo, bar): pass
    ...
    >>> breakfast = Foo('spam', 'eggs')
    >>> breakfast.foo, breakfast.bar
    ('spam', 'eggs')

    To restrict autoassignment to 'bar' and 'baz', write:

        @autoassign('bar', 'baz')
        def method(self, foo, bar, baz): ...

    To prevent 'foo' and 'baz' from being autoassigned, use:

        @autoassign(exclude=('foo', 'baz'))
        def method(self, foo, bar, baz): ...
    """
    if kwargs:
        exclude, f = set(kwargs['exclude']), None
        sieve = lambda l: ifilter(lambda nv: nv[0] not in exclude, l)
    elif len(names) == 1 and inspect.isfunction(names[0]):
        f = names[0]
        sieve = lambda l: l
    else:
        names, f = set(names), None
        sieve = lambda l: ifilter(lambda nv: nv[0] in names, l)

    def decorator(f):
        fargnames, _, _, fdefaults = inspect.getargspec(f)
        # Remove self from fargnames and make sure fdefault is a tuple
        fargnames, fdefaults = fargnames[1:], fdefaults or ()
        defaults = list(sieve(izip(reversed(fargnames), reversed(fdefaults))))

        @wraps(f)
        def decorated(self, *args, **kwargs):
            assigned = dict(sieve(izip(fargnames, args)))
            assigned.update(sieve(kwargs.iteritems()))
            for _ in starmap(assigned.setdefault, defaults):
                pass
            #self.__dict__.update(assigned)
            # better (more compatible):
            for k, v in assigned.iteritems():
                setattr(self, k, v)
            return f(self, *args, **kwargs)
        return decorated

    return f and decorator(f) or decorator


class GeneratorSpeed(pbar.ProgressBarWidget):
    def __init__(self):
        self.fmt = 'Speed: %d/s'
    def update(self, pbar):
        if pbar.seconds_elapsed < 2e-6:#== 0:
            bps = 0.0
        else:
            bps = float(pbar.currval) / pbar.seconds_elapsed
        return self.fmt % bps


class PBar(pbar.ProgressBar):
    def __init__(self, maxval):
        pbar.ProgressBar.__init__(self, widgets=[pbar.Percentage(), ' ',
                pbar.Bar(), ' ', pbar.ETA(), ' ', GeneratorSpeed()],
                maxval=maxval)

    def update(self, value=None):
        if value is None:
            pbar.ProgressBar.update(self, self.currval + 1)
        else:
            pbar.ProgressBar.update(self, value)

    def update(self, value=None):
        "Updates the progress bar to a new value."
        if value is None:
            value = self.currval + 1
        assert 0 <= value <= self.maxval
        self.currval = value
        if not self._need_update() or self.finished:
            return
        if not self.start_time:
            self.start_time = time.time()
        self.seconds_elapsed = time.time() - self.start_time
        self.prev_percentage = self.percentage()
        if value != self.maxval:
            self.fd.write(self._format_line() + '\r')
        else:
            self.finished = True
            self.fd.write(self._format_line() + '\r')


def norm(minimum, maximum, value):
    # return value
    if maximum == minimum:
        return maximum
    return (value - minimum) / (maximum - minimum)


def bounds(opt_w, opt_m, objective, zerobound=True):
    """
    Estimates the optimal and worst solution of the given scenario data.
    The zerobound parameter defines whether zero is assumed always. For the
    worst solution, an estimation based on the largest and smallest load
    profiles is performed.
    """
    w_max, w_min, w_max_r, w_min_r = (
            [None for i in range(opt_m)],
            [None for i in range(opt_m)],
            [None for i in range(opt_m)],
            [None for i in range(opt_m)])

    # Find minimal and maximal combination of elements
    keys = sorted(opt_w.keys())
    for i in range(len(opt_w)):
        w = opt_w[keys[i]]
        for j in range(len(w)):
            r = objective(w[j], record_call=False)
            if w_min_r[i] == None or r < w_min_r[i]:
                w_min[i] = w[j]
                w_min_r[i] = r
            if w_max_r[i] == None or r > w_max_r[i]:
                w_max[i] = w[j]
                w_max_r[i] = r

    # Assume 0 as optimal solution, select worst solution from (w_min,w_max)
    r_1 = objective(sum(w_max), record_call=False)
    r_2 = objective(sum(w_min), record_call=False)
    if zerobound:
        sol_d_max, sol_d_min = max(r_1, r_2), 0
    else:
        sol_d_max, sol_d_min = max(r_1, r_2), min(r_1, r_2)

    return sol_d_max, sol_d_min
