from os import path, makedirs, listdir, remove
from random import gauss, randint
import wget
from zipfile import ZipFile
import shutil
import warnings

import numpy as np
import pandas as pd

from simulator.workload.Workload import Workload
from simulator.container import IPSMBitbrain, RMBitbrain, DMBitbrain


warnings.simplefilter("ignore")

# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source:
# https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
ips_multiplier = 2054.0 / (2 * 600)


class BWGD(Workload):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        dataset_path = 'simulator/workload/datasets/bitbrain/'
        if not path.exists(dataset_path):
            makedirs(dataset_path)
            print('Downloading Bitbrain Dataset')
            url = 'http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip'
            filename = wget.download(url)
            zf = ZipFile(filename, 'r')
            zf.extractall(dataset_path)
            zf.close()
            for f in listdir(dataset_path + 'rnd/2013-9/'):
                shutil.move(dataset_path + 'rnd/2013-9/' + f, dataset_path + 'rnd/')
            shutil.rmtree(dataset_path + 'rnd/2013-7')
            shutil.rmtree(dataset_path + 'rnd/2013-8')
            shutil.rmtree(dataset_path + 'rnd/2013-9')
            remove(filename)
        self.dataset_path = dataset_path
        self.disk_sizes = [100, 200, 300, 400, 500]
        self.mean_sla, self.sigma_sla = 20, 3

    def generate_new_containers(self, interval):
        workload_list = []
        for i in range(np.random.poisson(self.lam)):
            creation_id = self.creation_id
            index = randint(1, 500)
            df = pd.read_csv(self.dataset_path + 'rnd/' + str(index) + '.csv', sep=';\t')
            sla = gauss(self.mean_sla, self.sigma_sla)
            ips_model = IPSMBitbrain((ips_multiplier * df['CPU usage [MHZ]']).to_list(),
                                     (ips_multiplier * df['CPU capacity provisioned [MHZ]']).to_list()[0],
                                     int(1.2 * sla), interval + sla)
            ram_model = RMBitbrain((df['Memory usage [KB]'] / 1000).to_list(),
                                   (df['Network received throughput [KB/s]'] / 1000).to_list(),
                                   (df['Network transmitted throughput [KB/s]'] / 1000).to_list())
            disk_size = self.disk_sizes[index % len(self.disk_sizes)]
            disk_model = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]'] / 1000).to_list(),
                                    (df['Disk write throughput [KB/s]'] / 1000).to_list())
            workload_list.append((creation_id, interval, ips_model, ram_model, disk_model))
            self.creation_id += 1
        self.created_containers += workload_list
        self.deployed_containers += [False] * len(workload_list)
        return self.get_undeployed_containers()
