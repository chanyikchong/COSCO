import numpy as np

from simulator.workload.Workload import Workload
from simulator.container import IPSMConstant, RMConstant, DMConstant


class SWSD(Workload):
    def __init__(self, num_workloads):
        super().__init__()
        self.num_workloads = num_workloads

    def generate_new_containers(self, interval):
        workload_list = []
        for i in range(self.num_workloads):
            creation_id = self.creation_id
            ips_multiplier = np.random.randint(1, 5)
            ips_model = IPSMConstant(1000 * ips_multiplier, 1500 * ips_multiplier, 4 * ips_multiplier,
                                    interval + 3 * ips_multiplier)
            ram_multiplier = np.random.randint(1, 5)
            ram_model = RMConstant(100 * ram_multiplier, 50 * ram_multiplier, 20 * ram_multiplier)
            disk_multiplier = np.random.randint(0, 1)
            disk_model = DMConstant(300 * disk_multiplier, 100 * disk_multiplier, 120 * disk_multiplier)
            workload_list.append((creation_id, interval, ips_model, ram_model, disk_model))
            self.creation_id += 1
        self.created_containers += workload_list
        self.deployed_containers += [False] * len(workload_list)
        return self.get_undeployed_containers()
