import sys

from .Scheduler import Scheduler
from .BaGTI.train import *


sys.path.append('scheduler/BaGTI/')


class GOBIScheduler(Scheduler):
    def __init__(self, data_type):
        super().__init__()
        self.model = eval(data_type + "()")
        self.model, _, _, _ = load_model(data_type, self.model, data_type)
        self.data_type = data_type
        self.hosts = int(data_type.split('_')[-1])
        dtl = data_type.split('_')
        _, _, self.max_container_ips = eval("load_" + '_'.join(dtl[:-1]) + "_data(" + dtl[-1] + ")")

    def _get_input_features(self):
        cpu = [host.getCPU() / 100 for host in self.env.hostlist]
        cpu = np.array([cpu]).transpose()
        if 'latency' in self.model.name:
            cpu_container = [(c.getApparentIPS() / self.max_container_ips if c else 0) for c in self.env.containerlist]
            cpu_container = np.array([cpu_container]).transpose()
            cpu = np.concatenate((cpu, cpu_container), axis=1)
        alloc = []
        prev_alloc = {}
        for c in self.env.containerlist:
            one_hot = [0] * len(self.env.hostlist)
            if c:
                prev_alloc[c.id] = c.getHostID()
            if c and c.getHostID() != -1:
                one_hot[c.getHostID()] = 1
            else:
                one_hot[np.random.randint(0, len(self.env.hostlist))] = 1
            alloc.append(one_hot)
        features = np.concatenate((cpu, alloc), axis=1)
        features = torch.tensor(features, dtype=torch.float, requires_grad=True)
        return features, prev_alloc

    def run_GOBI(self):
        init, prev_alloc = self._get_input_features()
        result, iteration, fitness = opt(init, self.model, [], self.data_type)
        decision = []
        for cid in prev_alloc:
            one_hot = result[cid, -self.hosts:].tolist()
            new_host = one_hot.index(max(one_hot))
            if prev_alloc[cid] != new_host:
                decision.append((cid, new_host))
        return decision

    def selection(self):
        return []

    def placement(self, containerIDs):
        first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
        decision = self.run_GOBI()
        return decision

    def allocation_fitness(self):
        init, prev_alloc = self._get_input_features()
        fitness = self.model.detail(init)
        return fitness
