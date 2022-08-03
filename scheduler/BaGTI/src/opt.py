import random
import torch
import numpy as np
from copy import deepcopy
from src.constants import *
from src.adahessian import Adahessian
import matplotlib.pyplot as plt


def convert_to_one_hot(data, cpu_old, hosts):
    alloc = []
    for i in data:
        one_hot = [0] * hosts
        alist = i.tolist()[-hosts:]
        one_hot[alist.index(max(alist))] = 1
        alloc.append(one_hot)
    new_data_one_hot = torch.cat((cpu_old, torch.FloatTensor(alloc)), dim=1)
    return new_data_one_hot


def opt(init, model, bounds, data_type):
    hosts = int(data_type.split('_')[-1])
    optimizer = torch.optim.AdamW([init], lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0
    equal = 0
    z_old = 100
    zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:, 0:-hosts])
        alloc_old = deepcopy(init.data[:, -hosts:])
        z = model(init)
        optimizer.zero_grad()
        z.backward()
        optimizer.step()
        scheduler.step()
        init.data = convert_to_one_hot(init.data, cpu_old, hosts)  # use old to keep the cpu feature
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:, -hosts:])) else 0
        if equal > 30:
            break
        iteration += 1
        z_old = z.item()
    #     zs.append(z.item())
    # plt.plot(zs); plt.show(); plt.clf()
    init.requires_grad = False
    return init.data, iteration, model(init)


def so_opt(init, model, bounds, data_type):
    hosts = int(data_type.split('_')[-1])
    optimizer = Adahessian([init], lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0
    equal = 0
    z_old = 100
    zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:, 0:-hosts])
        alloc_old = deepcopy(init.data[:, -hosts:])
        z = model(init)
        optimizer.zero_grad()
        z.backward(create_graph=True)
        optimizer.step()
        scheduler.step()
        init.data = convert_to_one_hot(init.data, cpu_old, hosts)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:, -hosts:])) else 0
        if equal > 30:
            break
        iteration += 1
        z_old = z.item()
    #     zs.append(z.item())
    # plt.plot(zs); plt.show(); plt.clf()
    init.requires_grad = False
    return init.data, iteration, model(init)
