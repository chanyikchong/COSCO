import numpy as np
import random

from .Workload import Workload


class AIoTW(Workload):
    def __init__(self, num_workloads, std_dev, database):
        super().__init__()
        self.num_workloads = num_workloads
        self.std_dev = std_dev
        self.db = database

    def generate_new_containers(self, interval):
        workload_list = []
        applications = ['resnet18', 'resnet34', 'squeezenet1_0', 'mobilenet_v2', 'mnasnet1_0', 'googlenet',
                        'resnext50_32x4d']
        multiplier = np.array([2, 1, 4, 2, 1, 3, 1])
        weights = 1 - (multiplier / np.sum(multiplier))
        for i in range(max(1, int(random.gauss(self.num_workloads, self.std_dev)))):
            creation_id = self.creation_id
            sla = np.random.randint(5, 8)  # Update this based on intervals taken
            application = random.choices(applications, weights=weights)[0]
            workload_list.append((creation_id, interval, sla, application))
            self.creation_id += 1
        self.created_containers += workload_list
        self.deployed_containers += [False] * len(workload_list)
        return self.get_undeployed_containers()
