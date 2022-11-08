import multiprocessing
import requests
import logging
import json

from joblib import Parallel, delayed
import platform

from metrics.powermodels import *  # import power models
from metrics.Disk import Disk
from metrics.RAM import RAM
from metrics.Bandwidth import Bandwidth
from utils.Utils import Color


num_cores = multiprocessing.cpu_count()


class Datacenter:
    def __init__(self, hosts, env, env_type):
        self.num_hosts = len(hosts)
        self.hosts = hosts
        self.env = env
        self.env_type = env_type
        self.types = {'Power': [1]}

    def parallelized_func(self, ip):
        payload = {"opcode": "hostDetails" + self.env_type}
        # todo how to get the information
        resp = requests.get("http://" + ip + ":8081/request", data=json.dumps(payload))
        data = json.loads(resp.text)
        return data

    def generate_hosts(self):
        print(Color.HEADER + "Obtaining host information and generating hosts" + Color.ENDC)
        hosts = []
        with open('framework/config/' + self.env + '_config.json', "r") as f:
            config = json.load(f)
        power_models = [server["powermodel"] for server in config[self.env.lower()]['servers']]
        # if self.env_type == 'Virtual':
        # -->
        with open('framework/server/scripts/instructions_arch.json') as f:
            arch_dict = json.load(f)
        instructions = arch_dict[platform.machine()]

        # get information of the host
        output_hosts_data = Parallel(n_jobs=num_cores)(delayed(self.parallelized_func)(i) for i in self.hosts)
        for i, data in enumerate(output_hosts_data):
            ip = self.hosts[i]
            logging.error("Host details collected from: {}".format(ip))
            print(Color.BOLD + ip + Color.ENDC, data)
            ips = (instructions * config[self.env.lower()]['servers'][i]['cpu']) / (
                    float(data['clock']) * 1000000) if self.env_type == 'Virtual' else data['MIPS']
            power = eval(power_models[i] + "()")
            ram = RAM(data['Total_Memory'], data['Ram_read'], data['Ram_write'])
            disk_ = Disk(data['Total_Disk'], data['Disk_read'], data['Disk_write'])
            bw = Bandwidth(data['Bandwidth'], data['Bandwidth'])
            hosts.append((ip, ips, ram, disk_, bw, power))
        return hosts
