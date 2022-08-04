from os import path, makedirs, listdir, remove
from random import gauss, randint
import wget
from zipfile import ZipFile
import shutil
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

from simulator.workload.Workload import Workload
from simulator.container import IPSMBitbrain, RMBitbrain, DMBitbrain


warnings.simplefilter("ignore")

# Intel Pentium III gives 2054 MIPS at 600 MHz
# Source:
# https://archive.vn/20130205075133/http://www.tomshardware.com/charts/cpu-charts-2004/Sandra-CPU-Dhrystone,449.html
ips_multiplier = 2054.0 / (2 * 600)


def create_files(df, year=2019):
    vmids = df[1].unique()[:1000].tolist()
    df = df[df[1].isin(vmids)]
    vmid = 0
    for i in tqdm(range(1, 501), ncols=80):
        trace = []
        bitbrain_df = pd.read_csv(f'simulator/workload/datasets/bitbrain/rnd/{i}.csv')
        req_len = len(bitbrain_df)
        while len(trace) < req_len:
            vmid = (vmid + 1) % len(vmids)
            trace += df[df[1] == vmids[vmid]][4].tolist()
        trace = trace[:req_len]
        pd.DataFrame(trace).to_csv(f'simulator/workload/datasets/azure_{str(year)}/{i}.csv', header=False, index=False)


class AzureWorkload(Workload):
    def __init__(self, lam, year=2019):
        super().__init__()
        self.lam = lam
        year = str(year)
        dataset_path = 'simulator/workload/datasets/bitbrain/'
        az_dpath = 'simulator/workload/datasets/azure_%s/' % year
        if not path.exists(dataset_path):
            makedirs(dataset_path)
            print('Downloading Bitbrain Dataset')
            url = 'http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/grid-workloads-archive/datasets/gwa-t-12/rnd.zip'
            url_alternate = 'https://www.dropbox.com/s/xk047xqcq9ue5hc/rnd.zip?dl=1'
            try:
                filename = wget.download(url)
            except:
                filename = wget.download(url_alternate)
            zf = ZipFile(filename, 'r')
            zf.extractall(dataset_path)
            zf.close()
            for f in listdir(dataset_path + 'rnd/2013-9/'):
                shutil.move(dataset_path + 'rnd/2013-9/' + f, dataset_path + 'rnd/')
            shutil.rmtree(dataset_path + 'rnd/2013-7')
            shutil.rmtree(dataset_path + 'rnd/2013-8')
            shutil.rmtree(dataset_path + 'rnd/2013-9')
            remove(filename)
        if not path.exists(az_dpath):
            makedirs(az_dpath)
            print('Downloading Azure %s Dataset' % year)
            if year == '2019':
                url = 'https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz'
            else:
                url = 'https://azurecloudpublicdataset.blob.core.windows.net/azurepublicdataset/trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-125.csv.gz'
            filename = wget.download(url)
            df = pd.read_csv(filename, header=None, compression='gzip')
            create_files(df)
            remove(filename)
        self.dataset_path = dataset_path
        self.az_dpath = az_dpath
        self.disk_sizes = [1, 2, 3]
        self.mean_sla, self.sigma_sla = 20, 3
        self.possible_indices = []
        for i in range(1, 500):
            df = pd.read_csv(self.dataset_path + 'rnd/' + str(i) + '.csv', sep=';\t')
            ips = (ips_multiplier * df['CPU usage [MHZ]']).to_list()[10]
            if 500 < ips < 3000:
                self.possible_indices.append(i)

    def generate_new_containers(self, interval):
        workload_list = []
        for i in range(np.random.poisson(self.lam)):
            creation_id = self.creation_id
            index = self.possible_indices[randint(0, len(self.possible_indices) - 1)]
            df = pd.read_csv(self.dataset_path + 'rnd/' + str(index) + '.csv', sep=';\t')
            df2 = pd.read_csv(self.az_dpath + str(index) + '.csv', header=None)
            sla = gauss(self.mean_sla, self.sigma_sla)
            ips = df['CPU capacity provisioned [MHZ]'].to_numpy() * df2.to_numpy()[:, 0] / 100
            ips_model = IPSMBitbrain((ips_multiplier * ips).tolist(),
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
