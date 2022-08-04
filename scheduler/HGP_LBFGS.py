import sys

from .Scheduler import Scheduler
from .HGP.train import *

sys.path.append('scheduler/HGP/')


class HGPScheduler(Scheduler):
    def __init__(self, data_type):
        super().__init__()
        self.model, _, _, self.max_container_ips = load_model(data_type)
        self.data_type = data_type
        self.hosts = int(data_type.split('_')[-1])

    def run_HGP(self):
        cpu = [host.get_cpu() / 100 for host in self.env.host_list]
        cpu = np.array([cpu]).transpose()
        cpu_container = [(c.get_apparent_ips() / self.max_container_ips if c else 0) for c in self.env.container_list]
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
        result, fitness = HGPopt(init, self.model, self.data_type)
        decision = []
        for cid in prev_alloc:
            one_hot = result[cid, -self.hosts:].tolist()
            new_host = one_hot.index(max(one_hot))
            if prev_alloc[cid] != new_host: decision.append((cid, new_host))
        return decision

    def selection(self):
        return []

    def placement(self, container_ids):
        first_alloc = np.all([not (c and c.get_host_id() != -1) for c in self.env.container_list])
        decision = self.run_HGP()
        return decision
