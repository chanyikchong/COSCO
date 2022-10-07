import pandas as pd
from random import gauss, randint

from simulator.workload.BitbrainWorkload2 import BWGD2
from simulator.container import IPSMBitbrain, RMBitbrain, DMBitbrain


ips_multiplier = 2054.0 / (2 * 600)


class MyWorkload(BWGD2):
    def __init__(self, arrival_rate, n_container):
        super(MyWorkload, self).__init__(arrival_rate)
        self.n_container = n_container

    def generate_new_containers(self, interval):
        workload_list = []
        for i in range(self.n_container):
            creation_id = self.creation_id
            index = self.possible_indices[:self.job_class_num][randint(0, self.job_class_num - 1)]
            df = pd.read_csv(self.dataset_path + 'rnd/' + str(index) + '.csv', sep=';\t')
            sla = gauss(self.mean_sla, self.sigma_sla)
            ips_model = IPSMBitbrain((ips_multiplier * df['CPU usage [MHZ]']).to_list(),
                                     (ips_multiplier * df['CPU capacity provisioned [MHZ]']).to_list()[0],
                                     int(1.2 * sla), interval + sla)
            ram_model = RMBitbrain((df['Memory usage [KB]'] / 4000).to_list(),
                                   (df['Network received throughput [KB/s]'] / 1000).to_list(),
                                   (df['Network transmitted throughput [KB/s]'] / 1000).to_list())
            disk_size = self.disk_sizes[index % len(self.disk_sizes)]
            disk_model = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]'] / 4000).to_list(),
                                    (df['Disk write throughput [KB/s]'] / 12000).to_list())
            workload_list.append((creation_id, interval, ips_model, ram_model, disk_model))
            self.creation_id += 1
        self.created_containers += workload_list
        self.deployed_containers += [False] * len(workload_list)
        return self.get_undeployed_containers()

    def set_n_container(self, n):
        self.n_container = n
