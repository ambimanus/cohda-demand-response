# coding=utf8

from __future__ import division

from cohda import util


class Configuration(object):

    @util.autoassign
    def __init__(self,
                 title='uvic',
                 seed=0,
                 basepath='data',
                 n=1500,
                 it=1441,
                 enviro='z7165526.txt',
                 lag=1,
                 ):
        pass