# coding=utf8

from __future__ import division

import sys
import os
import pickle

import numpy as np
# import scipy as sp
# import scipy.signal as sig
# import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib import ticker

import main


def resample(d, resolution):
    return d.reshape(d.shape[0]/resolution, resolution).sum(1) / resolution


def boxplot(ax, data, names, xl, yl, vert=True, rotate=False,
            xlabelpad=None, ylabelpad=None, xlim=None, ylim=None,
            xloc=None, xpos=None, logy=False):
    bp = ax.boxplot(data, vert=vert, positions=xpos)
    # bp = ax.boxplot(data)
    plt.setp(bp['boxes'], color='#1F4A7D', lw=0.5)
    plt.setp(bp['medians'], color='#1F4A7D', lw=1.0)
    plt.setp(bp['whiskers'], color='#1F4A7D', lw=0.5, ls='-')
    plt.setp(bp['caps'], color='#1F4A7D', lw=0.5)
    plt.setp(bp['fliers'], color='#1F4A7D', marker='+', markersize=3.0, markeredgewidth=0.5)
    # 348ABD : blue
    # 7A68A6 : purple
    # A60628 : red
    # 467821 : green
    # CF4457 : pink
    # 188487 : turquoise
    # E24A33 : orange
    #
    # Fill the boxes
    print yl
    for name, box, median in zip(names, bp['boxes'], bp['medians']):
        IQR = max(box.get_ydata()) - min(box.get_ydata())
        print '\t%s: median=%.2f, IQR=%.2f' % (name, median.get_ydata()[0], IQR)
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX,boxY)
        boxPolygon = Polygon(boxCoords, facecolor='#1F4A7D', alpha=0.25)
        ax.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(median.get_xdata()[j])
            medianY.append(median.get_ydata()[j])
            ax.plot(medianX, medianY, '#1F4A7D')
    #
    # if vert:
    #     ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
    #                   alpha=0.5)
    # else:
    #     ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
    #                   alpha=0.5)
    # Hide these grid behind plot objects
    # ax.set_axisbelow(True)
    ax.set_xlabel(xl, labelpad=xlabelpad)
    if logy:
        ax.set_yscale('log')
    if xpos is None:
        xpos = np.arange(1, int(len(names)) + 1)
    # have to loop over data to build averages due to possibly different data shapes
    averages = np.zeros(len(data))
    for i, d in enumerate(data):
        averages[i] = np.average(d)
    if vert:
        # Overplot the sample averages
        ax.plot(xpos, averages, linestyle='', color='#A60628', marker='*', markeredgewidth=0.0, markersize=4.5)
        # Set the axes ranges and axes labels
        ax.set_xlim(0.5, xpos[-1] + 0.5)
        # xticklabels = plt.setp(ax, xticklabels=names)
        if xloc is not None:
            ax.xaxis.set_major_locator(xloc)
        if rotate:
            plt.setp(xticklabels, fontsize=8, rotation=90)
    else:
        ax.plot(averages, xpos, linestyle='', color='#A60628', marker='*', markeredgewidth=0.0, markersize=4.5)
        ax.set_ylim(0.5, xpos[-1] + 0.5)
        # yticklabels = plt.setp(ax, yticklabels=names)
        if xloc is not None:
            ax.yaxis.set_major_locator(xloc)
        if rotate:
            plt.setp(yticklabels, fontsize=8, rotation=90)
    ax.set_ylabel(yl, labelpad=ylabelpad)
    # Set the axes ranges and axes labels
    if xlim is not None:
        if len(xlim) == 1:
            ax.set_xlim(left=xlim[0])
        else:
            ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        if len(ylim) == 1:
            ax.set_ylim(bottom=ylim[0])
        else:
            ax.set_ylim(ylim[0], ylim[1])


def _fluctuation_remaining(res_u, res_c):
    val_House = np.abs(res_c[1:] - res_c[:-1]).sum()
    val_House_u = np.abs(res_u[1:] - res_u[:-1]).sum()
    return (val_House * 100 / val_House_u) - 100


def _reduction(res_u, res_c):
    val_House = np.abs(res_c[1:] - res_c[:-1]).sum()
    val_House_u = np.abs(res_u[1:] - res_u[:-1]).sum()
    return 100 - (val_House * 100 / val_House_u)


def reduction(cfg):
    House = main.DotDict(cfg.House)
    House_uncontrolled = main.DotDict(cfg.House_uncontrolled)
    res_c = cfg.w - House.P_r.sum(0)
    res_u = cfg.w - House_uncontrolled.P_r.sum(0)
    return _reduction(res_u, res_c)


def reduction_multi(cfgs):
    red = np.zeros(len(cfgs))
    for i, cfg in enumerate(cfgs):
        red[i] = reduction(cfg)
    return red


def stats(reductions_dict):
    # import pdb
    # pdb.set_trace()
    # for k in sorted(reductions_dict.keys()):
    #     np.save(k, reductions_dict[k])
    keys = sorted(reductions_dict.keys())
    data = [reductions_dict[k] for k in keys]
    fig = plt.figure()
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.15, top=0.975)
    ax = fig.add_subplot(111)
    xpos = np.arange(len(keys))
    xlim = [-1, len(keys) + 1]
    # ax.plot(xpos, np.zeros(len(keys)), 'k')
    ax.hlines(0, xlim[0], xlim[1], linestyles='dotted')
    # xloc = ticker.FixedLocator(xpos)
    # xloc = ticker.MaxNLocator(nbins=6, prune=None)
    xloc = ticker.AutoLocator()
    boxplot(ax, data, keys, 'Delay [s]', '$\mathit{ROF}$ [\%]',
            xloc=xloc, xpos=xpos, xlim=xlim)
    plt.show()


def upsample(d, factor, it=None):
    if 'it' in d:
        if it is None:
            it = d['it']
        else:
            assert it == d['it'], (it, d['it'])
        d['it'] = it * factor
    assert it is not None
    for k in d.keys():
        v = d[k]
        if type(v) == dict:
            print k
            d[k] = upsample(v, factor, it=it)
        elif type(v) == np.ndarray:
            if v.shape[-1] == it:
                d[k] = np.repeat(v, factor, axis=-1)
    return d


def shift(arr, x, factor=60):
    assert x >= 0
    if x == 0:
        return arr
    arr = np.repeat(arr, factor, axis=-1)
    arr[x:] = arr[:-x]
    return resample(arr, factor)


def plot(cfg, s=0):
    # cfg.__dict__ = upsample(cfg.__dict__, 6)

    House = main.DotDict(cfg.House)
    House_uncontrolled = main.DotDict(cfg.House_uncontrolled)
    w = cfg.w
    it = cfg.it
    lag = cfg.lag

    bias = House.P_r.sum(0)[0] - w[0]

    follow = House.P_r.sum(0) - bias

    target = np.zeros(it)
    # no control actions in the first intervals defined by the lag
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

    diff = shift(diff, s)
    diff_pmin = shift(diff_pmin, s)
    diff_pmax = shift(diff_pmax, s)

    # Fast Fourier Transformation
    # signal = w - House.P_r.sum(0)
    # signal_u = w - House_uncontrolled.P_r.sum(0)
    # spacing = 60   # 1 sample per 60 seconds, yields Hz as frequencies
    # freq = fftpack.rfftfreq(it, d=spacing)
    # fft = abs(fftpack.rfft(signal))
    # fft_u = abs(fftpack.rfft(signal_u))
    # fig = plt.figure()
    # ax = fig.add_subplot(311)
    # ax.plot(signal, label='controlled')
    # ax.plot(signal_u, label='uncontrolled')
    # ax.legend()
    # ax = fig.add_subplot(312)
    # ax.plot(freq, 20 * sp.log10(fft), 'x')
    # ax = fig.add_subplot(313)
    # next(ax._get_lines.color_cycle) # skip a color in the cycle
    # ax.plot(freq, 20 * sp.log10(fft_u), 'x')
    # plt.show()

    # Savitzky-Golay
    # savgol = sig.savgol_filter(w - House_uncontrolled.P_r.sum(0), 71, 2)
    # val_House = np.abs(savgol - (w - House.P_r.sum(0))).sum()
    # val_House_u = np.abs(savgol - (w - House_uncontrolled.P_r.sum(0))).sum()

    # Gradient Residuals
    res_c = w - shift(House.P_r.sum(0), s)
    res_u = w - shift(House_uncontrolled.P_r.sum(0), s)

    red = _reduction(res_u, res_c)
    print 'Reduction: %.2f' % red
    print 'Messages: %.2f +- %.2f' % (np.mean(House.message_counter), np.std(House.message_counter))

    # Display the results
    fig = plt.figure(figsize=(6.39, 3.5))
    # fig.subplots_adjust(left=0.125, right=0.975, bottom=0.075, top=0.975)
    fig.subplots_adjust(left=0.125, right=0.975, bottom=0.125, top=0.975)
    # Resample the results to 15 minute resolution?
    res = 1

    ax = fig.add_subplot(311)
    ax.set_ylabel('P$_{\\mathrm{el}}$ [kW]')
    ax.set_ylim(-2000, 3000)
    # ax.set_ylim(-1000, 2000)
    l_w, = ax.plot(resample(w[1 : it], res), label='Wind Power', color='b', lw=0.25)
    # l_w.set_dashes([0.5, 0.5])
    l_w.set_dashes([1.0, 0.75])
    ax.fill_between(np.arange((it - 1) // res),
                    resample(diff_pmin[1: it], res),
                    resample(diff_pmax[1: it], res),
                     color=(0.5, 0.5, 0.5, 0.15), lw=0.0)
    # ax.plot(resample(f_pmin[1 : it], res), label='pmin')
    # ax.plot(resample(f_pmax[1 : it], res), label='pmax')
    fill_proxy = Rectangle((0, 0), 1, 1, fc=(0.5, 0.5, 0.5, 0.15), ec='w', lw=0.0)
    ax.plot(resample(diff[1 : it], res), label='Heat Pump Power Dispatched', color='k', lw=0.25)
    lhl = ax.get_legend_handles_labels()
    ax.legend(lhl[0] + [fill_proxy], lhl[1] + ['Capacity'], framealpha=0.5)
    plt.setp(ax.get_xticklabels(), visible=False)


    ax = fig.add_subplot(312, sharex=ax)
    # plt.setp(ax.spines.values(), color='k')
    # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='k')
    ax.set_ylabel('P$_{\\mathrm{el}}$ [kW]', labelpad=11)
    # ax.set_ylim(-350, 950)
    ax.plot(resample((diff - w)[1 : it], res), label='Error', color='k', lw=0.25)
    ax.legend(framealpha=0.5)
    plt.setp(ax.get_xticklabels(), visible=False)


    ax = fig.add_subplot(313, sharex=ax)
    # plt.setp(ax.spines.values(), color='k')
    # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='k')
    ax.set_ylabel('P$_{\\mathrm{el}}$ [kW]')
    # ax.set_ylim(-2000, 1500)
    # ax.plot(resample((diff - w)[1 : it], res), label='Error', color='k')
    # ax.fill_between(np.arange((it - 1) // res),
    #                 -1 * resample(House.Pmin[1 : it] + Target[1 : it], res),
    #                 -1 * resample(House.Pmax[1 : it] + Target[1 : it], res),
    #                  color=(0.5, 0.5, 0.5, 0.25), lw=0.0)
    l_res_u, = ax.plot(resample(res_u, res), label='resulting load (reference)', color='#7A68A6', lw=0.25)
    l_res_u.set_dashes([1.0, 0.75])
    ax.plot(resample(res_c, res), label='resulting load (controlled)', color='k', lw=0.25)
    # ax.plot(savgol, label='Savitzky-Golay')

    # lhl = ax.get_legend_handles_labels()
    # ax.legend(lhl[0] + [fill_proxy], lhl[1] + ['Flexibility'], loc='upper left')
    ax.legend(framealpha=0.5)

    ax.set_xlabel('Time [minutes]')


    # ax = fig.add_subplot(413, sharex=ax)
    # # plt.setp(ax.spines.values(), color='k')
    # # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='k')
    # # ax.set_xlabel('simulation horizon [minutes]')
    # ax.set_ylabel('T$_{\\mathrm{air}}^{\\mathrm{(reference)}}$ [\\textdegree{}C]')
    # ax.set_ylim(18.9, 21.1)
    # # ax.axhspan(19.75, 20.25, fc=(0.5, 0.5, 0.5, 0.2), ec=(1, 1, 1, 0))
    # for ts in House_uncontrolled.T_a[:, 1 : it]:
    #     ax.plot(resample(ts, res), color=(0.5, 0.5, 0.5, 0.1))


    # ax = fig.add_subplot(414, sharex=ax)
    # # plt.setp(ax.spines.values(), color='k')
    # # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='k')
    # ax.set_xlabel('Time [minutes]')
    # ax.set_ylabel('T$_{\\mathrm{air}}^{\\mathrm{(controlled)}}$ [\\textdegree{}C]')
    # ax.set_ylim(18.9, 21.1)
    # # ax.axhspan(19.75, 20.25, fc=(0.5, 0.5, 0.5, 0.2), ec=(1, 1, 1, 0))
    # for ts in House.T_a[:, 1 : it]:
    #     ax.plot(resample(ts, res), color=(0.5, 0.5, 0.5, 0.1))

    ax.set_xlim(0, 1440)
    # ax.set_xlim(790, 1210)
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
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            cfg = pickle.load(f)
        plot(cfg, s=10)

        # w = cfg.w
        # House = main.DotDict(cfg.House)
        # House_uncontrolled = main.DotDict(cfg.House_uncontrolled)
        # red_dict = {}
        # for s in range(61):
        #     res_c = w - shift(House.P_r.sum(0), s)
        #     res_u = w - shift(House_uncontrolled.P_r.sum(0), s)
        #     red_dict['%d s' % s] = _reduction(res_u, res_c)
        # stats(red_dict)

    else:
        # Calculate reductions for a several directories containing multiple
        # runs each, e.g.:
        # python analyze.py data/1000/lag*
        #
        # red_dict = {}
        # for fn in sys.argv[1:]:
        #     if fn[-1] == '/':
        #         fn = fn[:-1]
        #     if os.path.isdir(fn):
        #         dirname = os.path.basename(fn)
        #         files = sorted([os.path.join(fn, f)
        #                         for f in os.listdir(fn) if '.pickle' in f])
        #     else:
        #         dirname, filename = os.path.split(fn)
        #         files = [filename]
        #     reds = []
        #     for filename in files:
        #         with open(filename) as f:
        #             print filename
        #             reds.append(reduction(pickle.load(f)))
        #     red_dict[dirname] = reds
        # stats(red_dict)

        # Calculate reductions for multiple runs, and perform some artificial
        # time shifts for these runs, e.g.:
        # python analyze.py data/1000/lag00/*.pickle
        s_max = 60
        s_step = 1
        red_dict = {s: [] for s in range(0, s_max + 1, s_step)}
        for fn in sys.argv[1:]:
            with open(fn) as f:
                print fn
                cfg = pickle.load(f)
                w = cfg.w
                House = main.DotDict(cfg.House)
                House_uncontrolled = main.DotDict(cfg.House_uncontrolled)
                for s in range(0, s_max + 1, s_step):
                    res_c = w - shift(House.P_r.sum(0), s)
                    res_u = w - shift(House_uncontrolled.P_r.sum(0), s)
                    # red_dict[s].append(_fluctuation_remaining(res_u, res_c))
                    red_dict[s].append(_reduction(res_u, res_c))
        stats(red_dict)