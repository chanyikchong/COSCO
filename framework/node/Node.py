from datetime import datetime

from metrics.Disk import Disk
from metrics.RAM import RAM
from metrics.Bandwidth import Bandwidth


class Node:
    """
    Node work as a host
    # IPS = Million Instructions per second capacity
    # RAM = Ram in MB capacity
    # Disk = Disk characteristics capacity
    # Bw = Bandwidth characteristics capacity
    """
    def __init__(self, node_id, ip, ips, ram, disk, bw, power_model, framework):
        self.id = node_id
        self.ip = ip
        self.ips_cap = ips
        self.ram_cap = ram
        self.disk_cap = disk
        self.bw_cap = bw
        # Initialize utilization metrics
        self.ips = 0
        self.ram = RAM(0, 0, 0)
        self.bw = Bandwidth(0, 0)
        self.disk = Disk(0, 0, 0)
        self.json_body = {}
        self.power_model = power_model
        self.power_model.alloc_host(self)
        self.power_model.host = self
        self.env = framework
        self.create_host()

    def create_host(self):
        self.json_body = {
            "measurement": "host",
            "tags": {
                "host_id": self.id,
                "host_ip": self.ip
            },
            "time": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            "interval": self.env.interval,
            "fields": {
                "IPS_Cap": self.ips_cap,
                "RAM_Cap_size": self.ram_cap.size,
                "RAM_Cap_read": self.ram_cap.read,
                "RAM_Cap_write": self.ram_cap.write,
                "DISK_Cap_size": self.disk_cap.size,
                "DISK_Cap_read": self.disk_cap.read,
                "DISK_Cap_write": self.disk_cap.write,
                "Bw_Cap_up": self.bw_cap.uplink,
                "Bw_Cap_down": self.bw_cap.downlink,
                "IPS": self.ips,
                "RAM_size": self.ram.size,
                "RAM_read": self.ram.read,
                "RAM_write": self.ram.write,
                "DISK_size": self.disk.size,
                "DISK_read": self.disk.read,
                "DISK_write": self.disk.write,
                "Bw_up": self.bw.uplink,
                "Bw_down": self.bw.downlink,
                "Power": str(self.power_model.__class__.__name__)
            }
        }
        self.env.db.insert([self.json_body])

    def get_power(self):
        return self.power_model.power()

    def get_power_from_ips(self, ips):
        return self.power_model.power_from_cpu(min(100, 100 * (ips / self.ips_cap)))

    def get_cpu(self):
        # 0 - 100 last interval
        return min(100, 100 * (self.ips / self.ips_cap))

    def get_base_ips(self):
        return self.ips

    def get_apparent_ips(self):
        return self.ips

    def get_ips_available(self):
        return self.ips_cap - self.ips

    def get_current_ram(self):
        return self.ram.size, self.ram.read, self.ram.write

    def get_ram_available(self):
        size, read, write = self.get_current_ram()
        return max(0, (
            0.6 if self.ram_cap.size < 4000 else 0.8) * self.ram_cap.size - size), self.ram_cap.read - read, self.ram_cap.write - write

    def get_current_disk(self):
        return self.disk.size, self.disk.read, self.disk.write

    def get_disk_available(self):
        size, read, write = self.get_current_disk()
        return self.disk_cap.size - size, self.disk_cap.read - read, self.disk_cap.write - write

    def update_utilization_metrics(self):
        container_data, _ = self.env.controller.get_container_stat(self.ip)
        for container_d in container_data:
            ccid = int(container_d['fields']['name'].split("_")[0])
            container = self.env.get_container_by_cid(ccid)
            container.update_utilization_metrics(container_d['fields'])
        host_data, _ = self.env.controller.get_host_stat(self.ip)
        if 'fields' in host_data:
            self.ips = host_data['fields']['cpu'] * self.ips_cap / 100
            self.ram.size = host_data['fields']['memory']
            self.disk.size = host_data['fields']['disk']
        self.ram.read, self.ram.write = 0, 0
        self.disk.read, self.disk.write = 0, 0
        for cid in self.env.get_containers_of_host(self.id):
            self.ram.read += self.env.container_list[cid].ram.read
            self.disk.read += self.env.container_list[cid].disk.read
            self.ram.write += self.env.container_list[cid].ram.write
            self.disk.write += self.env.container_list[cid].ram.write
