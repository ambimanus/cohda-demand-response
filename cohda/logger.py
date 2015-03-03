# coding=utf-8

import logging

from objectives import Objective

FORMAT = '[%(levelname)-9s] %(message)s'

_LVL_AGENTV = 11
_LVL_AGENT = 12
_LVL_STATS = 21
_LVL_SOLUTION = 22
_LVL_TARGET = 23

_LVL_AGENTV_NAME = 'AGENTV'
_LVL_AGENT_NAME = 'AGENT'
_LVL_STATS_NAME = 'STATS'
_LVL_SOLUTION_NAME = 'SOLUTION'
_LVL_TARGET_NAME = 'TARGET'

# LOG_LEVEL = logging.INFO
# LOG_LEVEL = _LVL_SOLUTION
LOG_LEVEL = 1000

FILTER = None
#FILTER = ('bkc','BKC')
FILTER_LVL = None
#FILTER_LVL = _LVL_SOLUTION


def setup_logger(lvl=LOG_LEVEL):
    if not '_logger' in globals():
        logging.addLevelName(_LVL_AGENT, _LVL_AGENT_NAME)
        logging.addLevelName(_LVL_AGENTV, _LVL_AGENTV_NAME)
        logging.addLevelName(_LVL_STATS, _LVL_STATS_NAME)
        logging.addLevelName(_LVL_SOLUTION, _LVL_SOLUTION_NAME)
        logging.addLevelName(_LVL_TARGET, _LVL_TARGET_NAME)
        logging.basicConfig(level=lvl, format=FORMAT)
        globals()['_logger'] = logging.getLogger('crystal')

    globals()['message_counter'] = 0
    globals()['first_time'] = True


def reset_message_counter():
    globals()['message_counter'] = 0


def MSG():
    globals()['message_counter'] += 1


def log(lvl, *msg):
    if LOG_LEVEL >= 1000:
        return
    message = _string(*msg)
    if (lvl == FILTER_LVL or
            FILTER == None or any([True for f in FILTER if f in message])):
        _logger.log(lvl, message)


def _string(*msg):
    if globals()['first_time']:
        out = '  msg |   obj | '
        globals()['first_time'] = False
    else:
        out = '%5d | %5d | ' % (message_counter, Objective.calls)
    if len(msg) > 1:
        if type(msg[0] == int or (type(msg[0] == str and len(msg[0]) > 0))):
            sep = ' | '
        else:
            sep = ' '
        if type(msg[0]) == int:
            out += 'a%04d' % msg[0] + sep
        else:
            out += str(msg[0]) + sep
        msg = msg[1:]
    for s in msg:
        out += str(s) + ' '
    return out


def AGENTV(*msg):
    log(_LVL_AGENTV, *msg)


def AGENT(*msg):
    log(_LVL_AGENT, *msg)


def STATS(*msg):
    log(_LVL_STATS, *msg)


def SOLUTION(*msg):
    log(_LVL_SOLUTION, *msg)


def DEBUG(*msg):
    log(logging.DEBUG, *msg)


def INFO(*msg):
    log(logging.INFO, *msg)


def WARNING(*msg):
    log(logging.WARNING, *msg)


def ERROR(*msg):
    log(logging.ERROR, *msg)
