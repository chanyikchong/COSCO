import sys

from .Scheduler import Scheduler
from .BaGTI.train import *
from .BaGTI.src.utils import *

sys.path.append('scheduler/BaGTI/')


class SOGOBI2Scheduler(Scheduler):
    def __init__(self, data_type):
        super().__init__()
        dtl = data_type.split('_')
        data_type = '_'.join(dtl[:-1]) + '2_' + dtl[-1]
        self.model = eval(data_type + "()")
        self.model, _, _, _ = load_model(data_type, self.model, data_type)
        self.data_type = data_type
        self.hosts = int(data_type.split('_')[-1])
        _, _, (self.max_container_ips, self.max_energy, self.max_response) = eval(
            "load_" + '_'.join(dtl[:-1]) + "2_data(" + dtl[-1] + ")")

    def run_SOGOBI2(self):
        cpu = [host.get_cpu() / 100 for host in self.env.host_list]
        cpu = np.array([cpu]).transpose()
        if 'latency' in self.model.name:
            cpu_container = [(c.get_apparent_ips() / self.max_container_ips if c else 0) for c in
                             self.env.container_list]
            cpu_container = np.array([cpu_container]).transpose()
            e, r = (0, 0) if self.env.stats is None else self.env.stats.run_simulation_GOBI()
            pred = np.broadcast_to(np.array([e / self.max_energy, r / self.max_response]), (self.hosts, 2))
            cpu = np.concatenate((cpu, cpu_container, pred), axis=1)
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
        result, iteration, fitness = so_opt(init, self.model, [], self.data_type)
        decision = []
        for cid in prev_alloc:
            one_hot = result[cid, -self.hosts:].tolist()
            new_host = one_hot.index(max(one_hot))
            if prev_alloc[cid] != new_host:
                decision.append((cid, new_host))
        return decision

    def selection(self):
        return []

    def placement(self, container_ids):
        first_alloc = np.all([not (c and c.get_host_id() != -1) for c in self.env.container_list])
        decision = self.run_SOGOBI2()
        return decision
