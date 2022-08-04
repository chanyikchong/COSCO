import simulator.host as host
import metrics.powermodels as pm


class AzureFog:
    def __init__(self, num_hosts):
        self.num_hosts = num_hosts
        self.edge_hosts = round(num_hosts * 0.6)
        self.types = {
            'B2s':
                {
                    'IPS': 4029,
                    'RAMSize': 4295,
                    'RAMRead': 372.0,
                    'RAMWrite': 200.0,
                    'DiskSize': 32212,
                    'DiskRead': 13.42,
                    'DiskWrite': 1.011,
                    'BwUp': 5000,
                    'BwDown': 5000,
                    'Power': 'PMB2s'
                },
            'B4ms':
                {
                    'IPS': 4029,
                    'RAMSize': 17180,
                    'RAMRead': 360.0,
                    'RAMWrite': 305.0,
                    'DiskSize': 32212,
                    'DiskRead': 10.38,
                    'DiskWrite': 0.619,
                    'BwUp': 5000,
                    'BwDown': 5000,
                    'Power': 'PMB4ms'
                },
            'B8ms':
                {
                    'IPS': 16111,
                    'RAMSize': 34360,
                    'RAMRead': 376.54,
                    'RAMWrite': 266.75,
                    'DiskSize': 32212,
                    'DiskRead': 11.64,
                    'DiskWrite': 1.164,
                    'BwUp': 5000,
                    'BwDown': 5000,
                    'Power': 'PMB8ms'
                }
        }

    def generate_hosts(self):
        hosts = []
        types = ['B2s', 'B2s', 'B2s', 'B2s', 'B4ms', 'B4ms', 'B4ms', 'B4ms', 'B8ms', 'B8ms'] * 5
        for i in range(self.num_hosts):
            type_id = types[i]
            ips = self.types[type_id]['IPS']
            ram = host.RAM(self.types[type_id]['RAMSize'], self.types[type_id]['RAMRead'] * 5,
                           self.types[type_id]['RAMWrite'] * 5)
            disk = host.Disk(self.types[type_id]['DiskSize'], self.types[type_id]['DiskRead'] * 5,
                             self.types[type_id]['DiskWrite'] * 10)
            bw = host.Bandwidth(self.types[type_id]['BwUp'], self.types[type_id]['BwDown'])
            power = eval("pm.%s()" % self.types[type_id]['Power'])
            latency = 0.003 if i < self.edge_hosts else 0.076
            hosts.append((ips, ram, disk, bw, latency, power))
        return hosts
