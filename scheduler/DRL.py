from os import path, mkdir
import pickle
from copy import deepcopy
from random import sample, randint

# import numpy as np
# import torch

from .A2C.rl import *
# from .A2C.rl import backprop, save_model, plot_accuracies, load_model, Coeff_Energy, Coeff_Latency
from .Scheduler import Scheduler


class DRLScheduler(Scheduler):
    def __init__(self, data_type, training=True):
        super().__init__()
        self.model = eval(data_type + "_RL()")
        self.training = training
        self.hosts = int(data_type.split('_')[-1])
        self.last_schedule = None
        self.model, self.optimizer, self.epoch, self.accuracy_list = load_model(data_type + '_RL', self.model,
                                                                                data_type)
        self.data_type = data_type
        self.buffer = []
        directory = 'scheduler/A2C/dataset/'
        self.buffer_path = directory + 'buffer.pkl'
        if not path.exists(directory): mkdir(directory)
        if path.exists(self.buffer_path):
            with open(self.buffer_path, "rb") as handle:
                self.buffer = pickle.load(handle)

    def get_current_schedule(self):
        cpu = [host.get_cpu() / 100 for host in self.env.host_list]
        cpu = np.array([cpu]).transpose()
        if '_' in self.data_type:
            cpu_container = [(c.get_apparent_ips() / 6000 if c else 0) for c in self.env.container_list]
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
        init = torch.tensor(init, dtype=torch.float)
        return init, prev_alloc

    def get_last_value(self):
        energy = self.env.stats.metrics[-1]['energytotalinterval']
        all_energies = [e['energytotalinterval'] for e in self.env.stats.metrics[-20:]]
        all_latencies = [e['avgresponsetime'] for e in self.env.stats.metrics[-20:]]
        min_e, max_e = min(all_energies), max(all_energies)
        energy = (energy - min_e) / (0.1 + max_e - min_e)
        if '_' in self.data_type:
            latency = self.env.stats.metrics[-1]['avgresponsetime']
            latency = latency / (0.1 + max(all_latencies))
            return Coeff_Energy * energy + Coeff_Latency * latency
        return energy

    def run_DRL(self, container_ids, rnd):
        self.epoch += 1
        schedule_next, prev_alloc = self.get_current_schedule()
        value_t = self.get_last_value() if self.last_schedule else 0.5
        schedule_t = self.last_schedule if self.last_schedule else schedule_next
        if self.training and len(self.buffer) >= 10:
            for replay in sample(self.buffer, 10):
                backprop(replay[0], replay[1], replay[2], replay[3], self.optimizer)
        vl, pl, action = backprop(self.model, schedule_t, value_t, schedule_next, self.optimizer)
        if rnd:
            decision = self.random_placement(container_ids)
            schedule_next = deepcopy(schedule_t)
            for cid, hid in decision:
                schedule_next[cid][hid + 2] = 1
                schedule_next[cid][prev_alloc[cid] + 2] = 0
            self.buffer.append((self.model, schedule_t, value_t, schedule_next))
        else:
            self.buffer.append((self.model, schedule_t, value_t, schedule_next))
            decision = []
            for cid in prev_alloc:
                one_hot = action.tolist()
                new_host = one_hot.index(max(one_hot))
                if prev_alloc[cid] != new_host: decision.append((cid, new_host))
        self.buffer = self.buffer[-50:]
        self.accuracy_list.append((vl + pl, vl, pl))
        print(vl, pl)
        self.last_schedule = schedule_t
        if self.training and self.epoch % 10 == 0:
            save_model(self.model, self.optimizer, self.epoch, self.accuracy_list)
            with open(self.buffer_path, "wb") as handle:
                pickle.dump(self.buffer, handle)
            plot_accuracies(self.accuracy_list)
        return decision

    def selection(self):
        return []

    def placement(self, container_ids):
        # first_alloc = np.all([not (c and c.get_host_id() != -1) for c in self.env.container_list])
        decision = self.run_DRL(container_ids, self.training and randint(0, 100) < 30)
        return decision
