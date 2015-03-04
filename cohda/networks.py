# coding=utf-8

from __future__ import division

import copy


def unconnected(cfg):
    network = {}
    for n in cfg.agent_ids:
        network[n] = list()
    return network


def sequence(cfg, network=None):
    ids = cfg.agent_ids
    if network is None:
        network = unconnected(cfg)
    for i in range(len(ids)):
        for k in range(1, cfg.network_k + 1):
            if i + k < len(ids):
                network[ids[i]].append(ids[i + k])
                network[ids[i + k]].append(ids[i])
    return network


def ring(cfg, network=None):
    ids = cfg.agent_ids
    if network is None:
        network = unconnected(cfg)
    for i in range(len(ids)):
        for k in range(1, min(len(ids), cfg.network_k + 1)):
            node = (i + k) % len(ids)
            network[ids[i]].append(ids[node])
            network[ids[node]].append(ids[i])
    return network


def full(cfg, network=None):
    ids = cfg.agent_ids
    if network is None:
        network = {}
    for i in range(len(ids)):
        network[ids[i]] = [ids[x] for x in range(len(ids)) if x != i]
    return network


def half(cfg, network=None):
    ids = cfg.agent_ids
    if network is None:
        network = unconnected(cfg)
    for n1 in ids[:len(ids)/2]:
        for n2 in ids[len(ids)/2:]:
            network[n1].append(n2)
            network[n2].append(n1)
    return network


def mesh_rect(cfg, network=None):
    from math import floor, sqrt
    ids = cfg.agent_ids
    if network is None:
        network = {}
    s = int(floor(sqrt(len(ids))))
    for i in range(len(ids)):
        d = list()
        if i - s >= 0:
            d.append(ids[i - s])      # Node above i
        if i + s < len(ids):
            d.append(ids[i + s])      # Node below i
        if i % s > 0 and i > 0:
            d.append(ids[i - 1])      # Node left from i
        if (i + 1) % s > 0 and i + 1 < len(ids):
            d.append(ids[i + 1])      # Node right from i
        network[ids[i]] = d
    return network


def random(cfg, network=None):
    ids, rnd = cfg.agent_ids, cfg.rnd
    if network is None:
        sids = list(ids)
        rnd.shuffle(sids)
        cfgs = copy.copy(cfg)
        cfgs.agent_ids = sids
        network = sequence(cfgs)
    c = cfg.network_c
    for n in ids:
        if len(network[n]) >= c:
            continue
        cand = list(ids)
        rnd.shuffle(cand)
        for cnd in cand:
            if len(network[n]) >= c:
                break
            if (cnd == n or cnd in network[n] or
                    len(network[cnd]) >= c):
                continue
            network[n].append(cnd)
            network[cnd].append(n)
    return network


def smallworld(cfg, network=None):
    # First create a k-neighbour-ring
    network = ring(cfg, network=network)
    # Create len(ids)*k*phi random shortcuts
    ids, k, phi = cfg.agent_ids, cfg.network_k, cfg.network_phi
    for i in range(int(len(ids) * k * phi)):
        subset = cfg.rnd.sample(ids, 2)
        if not subset[1] in network[subset[0]]:
            network[subset[0]].append(subset[1])
            network[subset[1]].append(subset[0])
    return network

