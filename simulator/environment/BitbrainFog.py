import simulator.host as host
import metrics.powermodels as pm


class BitbrainFog:
    def __init__(self, num_hosts):
        self.num_hosts = num_hosts
        self.types = {
            'IPS': [5000, 10000, 50000],  # Some containers have IPS requirement as high as 40k
            'RAMSize': [3000, 4000, 8000],
            'RAMRead': [3000, 2000, 3000],
            'RAMWrite': [3000, 2000, 3000],
            'DiskSize': [30000, 40000, 80000],
            'DiskRead': [2000, 2000, 3000],
            'DiskWrite': [2000, 2000, 3000],
            'BwUp': [1000, 2000, 5000],
            'BwDown': [2000, 4000, 10000],
            'Power': [1]
        }

    def generate_hosts(self):
        hosts = []
        for i in range(self.num_hosts):
            type_id = i % 3  # np.random.randint(0,3) # i%3 #
            ips = self.types['IPS'][type_id]
            ram = host.RAM(self.types['RAMSize'][type_id], self.types['RAMRead'][type_id],
                           self.types['RAMWrite'][type_id])
            disk = host.Disk(self.types['DiskSize'][type_id], self.types['DiskRead'][type_id],
                             self.types['DiskWrite'][type_id])
            bw = host.Bandwidth(self.types['BwUp'][type_id], self.types['BwDown'][type_id])
            power = pm.PMConstant(self.types['Power'][type_id]) if type_id < 1 else pm.PMRaspberryPi()
            hosts.append((ips, ram, disk, bw, 0, power))
        return hosts
