import numpy as np
import random

from .Workload import Workload


class DFW(Workload):
    def __init__(self, num_workloads, std_dev, database):
        super().__init__()
        self.num_workloads = num_workloads
        self.std_dev = std_dev
        self.db = database

    def generate_new_containers(self, interval):
        workload_list = list()
        applications = ['shreshthtuli/yolo', 'shreshthtuli/pocketsphinx', 'shreshthtuli/aeneas']
        for i in range(max(1, int(random.gauss(self.num_workloads, self.std_dev)))):
            creation_id = self.creation_id
            sla = np.random.randint(5, 8)  # Update this based on intervals taken
            application = random.choices(applications, weights=[0.2, 0.4, 0.4])[0]
            workload_list.append((creation_id, interval, sla, application))
            self.creation_id += 1
        self.created_containers += workload_list
        self.deployed_containers += [False] * len(workload_list)
        return self.get_undeployed_containers()
