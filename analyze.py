# coding=utf8

from __future__ import division

import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import main


def plot(cfg):
    House = main.DotDict(cfg.House)
    House_uncontrolled = main.DotDict(cfg.House_uncontrolled)
    w = cfg.w
    it = cfg.it
    lag = cfg.lag

    # Resample the results to 15 minute resolution?
    res = 1
    def resample(d, resolution):
        return d.reshape(d.shape[0]/resolution, resolution).sum(1) / resolution

    # Display the results
    fig = plt.figure()

    # bias_P_r = House.P_r.sum(0)[0]
    # bias_wind = w[0]
    bias = House.P_r.sum(0)[0] - w[0]

    follow = House.P_r.sum(0) - bias

    target = np.zeros(it)
    # no control actions in the first intervals defined by the lag
    # target[:2] = House.P_r[:,:2].sum(0) - bias
    # target[2:] = House.P_target[2:] - bias
    target[:lag] = House.P_r[:,:lag].sum(0) - bias
    target[lag:] = House.P_target[lag:] - bias

    # f_pmin, f_pmax = np.zeros(it), np.zeros(it)
    f_pmin = House.Pmin - bias
    f_pmax = House.Pmax - bias


    diff = np.zeros(it)
    diff[lag:] = np.array(follow - target + w)[:it - lag]
    diff_pmin, diff_pmax = np.array(f_pmin), np.array(f_pmax)
    diff_pmin[lag + 1:] = np.array(f_pmin - target + w)[1:it - lag]
    diff_pmax[lag + 1:] = np.array(f_pmax - target + w)[1:it - lag]
    # JAY!!!

    ax = fig.add_subplot(411)
    ax.set_ylabel('P$_{\\mathrm{el}}$ [kW]')
    ax.plot(resample(w[1 : it], res), label='Wind Power', lw=0.5)
    ax.fill_between(np.arange((it - 1) // res),
                    resample(diff_pmin[1: it], res),
                    resample(diff_pmax[1: it], res),
                     color=(0.5, 0.5, 0.5, 0.25), lw=0.0)
    # ax.plot(resample(f_pmin[1 : it], res), label='pmin')
    # ax.plot(resample(f_pmax[1 : it], res), label='pmax')
    fill_proxy = Rectangle((0, 0), 1, 1, fc=(0.5, 0.5, 0.5, 0.25), ec='w', lw=0.0)
    ax.plot(resample(diff[1 : it], res), label='Heat Pump Power Dispatched', color='k')
    lhl = ax.get_legend_handles_labels()
    ax.legend(lhl[0] + [fill_proxy], lhl[1] + ['Capacity'], framealpha=0.5)


    ax = fig.add_subplot(412, sharex=ax)
    # plt.setp(ax.spines.values(), color='k')
    # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='k')
    ax.set_ylabel('P$_{\\mathrm{el}}$ [kW]')
    # ax.plot(resample((diff - w)[1 : it], res), label='Error', color='k')
    # ax.fill_between(np.arange((it - 1) // res),
    #                 -1 * resample(House.Pmin[1 : it] + Target[1 : it], res),
    #                 -1 * resample(House.Pmax[1 : it] + Target[1 : it], res),
    #                  color=(0.5, 0.5, 0.5, 0.25), lw=0.0)
    ax.plot(resample(w - House_uncontrolled.P_r.sum(0), res), label='resulting load (reference)')
    ax.plot(resample(w - House.P_r.sum(0), res), label='resulting load (controlled)')

    # lhl = ax.get_legend_handles_labels()
    # ax.legend(lhl[0] + [fill_proxy], lhl[1] + ['Flexibility'], loc='upper left')
    ax.legend(framealpha=0.5)


    ax = fig.add_subplot(413, sharex=ax)
    # plt.setp(ax.spines.values(), color='k')
    # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='k')
    # ax.set_xlabel('simulation horizon [minutes]')
    ax.set_ylabel('T$_{\\mathrm{air}}^{\\mathrm{(reference)}}$ [\\textdegree{}C]')
    ax.set_ylim(18.9, 21.1)
    # ax.axhspan(19.75, 20.25, fc=(0.5, 0.5, 0.5, 0.2), ec=(1, 1, 1, 0))
    for ts in House_uncontrolled.T_a[:, 1 : it]:
        ax.plot(resample(ts, res), color=(0.5, 0.5, 0.5, 0.1))


    ax = fig.add_subplot(414, sharex=ax)
    # plt.setp(ax.spines.values(), color='k')
    # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='k')
    # ax.set_xlabel('simulation horizon [minutes]')
    ax.set_ylabel('T$_{\\mathrm{air}}^{\\mathrm{(controlled)}}$ [\\textdegree{}C]')
    ax.set_ylim(18.9, 21.1)
    # ax.axhspan(19.75, 20.25, fc=(0.5, 0.5, 0.5, 0.2), ec=(1, 1, 1, 0))
    for ts in House.T_a[:, 1 : it]:
        ax.plot(resample(ts, res), color=(0.5, 0.5, 0.5, 0.1))


    # ax = fig.add_subplot(414, sharex=ax)
    # # plt.setp(ax.spines.values(), color='k')
    # # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='k')
    # ax.set_xlabel('simulation horizon [minutes]')
    # ax.set_ylabel('messages')
    # # ax.set_ylim(18.9, 21.1)
    # # ax.axhspan(19.75, 20.25, fc=(0.5, 0.5, 0.5, 0.2), ec=(1, 1, 1, 0))
    # ax.plot(resample(House.message_counter[1 : it], res))

    # plt.ion()
    # plt.show()
    # import pdb
    # pdb.set_trace()
    plt.show()


if __name__ == '__main__':
    assert len(sys.argv) == 2
    with open(sys.argv[1]) as f:
        cfg = pickle.load(f)
    plot(cfg)
