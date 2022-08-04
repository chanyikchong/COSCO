import sys

# import numpy as np
# import torch

from .Scheduler import Scheduler
from .BaGTI.train import *

sys.path.append('scheduler/BaGTI/')


class GAScheduler(Scheduler):
    def __init__(self, data_type):
        super().__init__()
        self.model = eval(data_type + "()")
        self.model, _, _, _ = load_model(data_type, self.model, data_type)
        self.hosts = int(data_type.split('_')[-1])
        self.data_type = data_type
        dtl = data_type.split('_')
        self.dataset, self.dataset_size, self.max_container_ips = eval(
            "load_" + '_'.join(dtl[:-1]) + "_data(" + dtl[-1] + ")")

    def run_GA(self):
        cpu = [host.get_cpu() / 100 for host in self.env.host_list]
        cpu = np.array([cpu]).transpose()
        if 'latency' in self.model.name:
            cpu_container = [(c.get_apparent_ips() / self.max_container_ips if c else 0) for c in
                             self.env.container_list]
            cpu_container = np.array([cpu_container]).transpose()
            cpu = np.concatenate((cpu, cpu_container), axis=1)
        alloc = []
        prev_alloc = {}
        for c in self.env.container_list:
            one_hot = [0] * len(self.env.host_list)
            if c:
                prev_alloc[c.id] = c.get_host_id()
            if c and c.get_host_id() != -1:
                one_hot[c.get_host_id()] = 1
            else:
                one_hot[np.random.randint(0, len(self.env.host_list))] = 1
            alloc.append(one_hot)
        init = np.concatenate((cpu, alloc), axis=1)
        init = torch.tensor(init, dtype=torch.float, requires_grad=True)
        result, iteration, fitness = ga(self.dataset, self.model, [], self.data_type, self.hosts)
        decision = []
        for cid in prev_alloc:
            one_hot = result[cid, (2 if 'latency' in self.model.name else 1):].tolist()
            new_host = one_hot.index(max(one_hot))
            if prev_alloc[cid] != new_host:
                decision.append((cid, new_host))
        return decision

    def selection(self):
        return []

    def placement(self, container_ids):
        first_alloc = np.all([not (c and c.get_host_id() != -1) for c in self.env.container_list])
        decision = self.run_GA()
        return decision
