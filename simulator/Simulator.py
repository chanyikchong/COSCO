import numpy as np

from simulator.host import Host
from simulator.container import Container


class Simulator:
    # Total power in watt
    # Total Router Bw
    # Interval Time in seconds
    def __init__(self, total_power, router_bw, scheduler, container_limit, interval_time, host_init):
        self.total_power = total_power
        self.total_bw = router_bw
        self.host_limit = len(host_init)
        self.scheduler = scheduler
        self.scheduler.set_environment(self)
        self.container_limit = container_limit
        self.host_list = list()
        self.container_list = list()
        self.interval_time = interval_time
        self.interval = 0
        self.inactive_containers = list()
        self.stats = None
        self.add_host_list_init(host_init)

    def add_host_init(self, ips, ram, disk, bw, latency, power_model):
        assert len(self.host_list) < self.host_limit
        host = Host(len(self.host_list), ips, ram, disk, bw, latency, power_model, self)
        self.host_list.append(host)

    def add_host_list_init(self, host_list):
        assert len(host_list) == self.host_limit
        for ips, ram, disk, bw, latency, power_model in host_list:
            self.add_host_init(ips, ram, disk, bw, latency, power_model)

    def add_container_init(self, creation_id, creation_interval, ips_model, ram_model, disk_model):
        container = Container(len(self.container_list), creation_id, creation_interval, ips_model, ram_model,
                              disk_model, self, host_id=-1)
        self.container_list.append(container)
        return container

    def add_container_list_init(self, container_info_list):
        deployed = container_info_list[
                   :min(len(container_info_list), self.container_limit - self.get_num_active_containers())]
        deployed_containers = list()
        for creation_id, creation_interval, ips_model, ram_model, disk_model in deployed:
            dep = self.add_container_init(creation_id, creation_interval, ips_model, ram_model, disk_model)
            deployed_containers.append(dep)
        self.container_list += [None] * (self.container_limit - len(self.container_list))
        return [container.id for container in deployed_containers]

    def add_container(self, creation_id, creation_interval, ips_model, ram_model, disk_model):
        for i, c in enumerate(self.container_list):
            if c is None or not c.active:
                container = Container(i, creation_id, creation_interval, ips_model, ram_model, disk_model, self,
                                      host_id=-1)
                self.container_list[i] = container
                return container

    def add_container_list(self, container_info_list):
        deployed = container_info_list[
                   :min(len(container_info_list), self.container_limit - self.get_num_active_containers())]
        deployed_containers = list()
        for creation_id, creation_interval, ips_model, ram_model, disk_model in deployed:
            dep = self.add_container(creation_id, creation_interval, ips_model, ram_model, disk_model)
            deployed_containers.append(dep)
        return [container.id for container in deployed_containers]

    def get_containers_of_host(self, host_id):
        containers = []
        for container in self.container_list:
            if container and container.host_id == host_id:
                containers.append(container.id)
        return containers

    def get_container_by_id(self, container_id):
        return self.container_list[container_id]

    def get_container_by_cid(self, creation_id):
        for c in self.container_list + self.inactive_containers:
            if c and c.creation_id == creation_id:
                return c

    def get_host_by_id(self, host_id):
        # todo return last host when hostID = -1 BUG
        return self.host_list[host_id]

    def get_creation_ids(self, migrations, container_ids):
        creation_ids = list()
        for decision in migrations:
            if decision[0] in container_ids:
                creation_ids.append(self.container_list[decision[0]].creation_id)
        return creation_ids

    def get_placement_possible(self, container_id, host_id):
        container = self.container_list[container_id]
        host = self.host_list[host_id]
        ips_req = container.get_base_ips()
        ram_size_req, ram_read_req, ram_write_req = container.get_ram()
        disk_size_req, disk_read_req, disk_write_req = container.get_disk()
        ips_available = host.get_ips_available()
        ram_size_avil, ram_read_avil, ram_write_avil = host.get_ram_available()
        disk_size_avil, disk_read_avil, disk_write_avil = host.get_disk_available()
        return (ips_req <= ips_available and
                ram_size_req <= ram_size_avil and
                # ram_read_req <= ram_read_avil and
                # ram_write_req <= ram_write_avil and
                disk_size_req <= disk_size_avil
                # disk_read_req <= disk_read_avil and
                # disk_write_req <= disk_write_avil
                )

    def add_containers_init(self, container_info_list_init):
        self.interval += 1
        deployed = self.add_container_list_init(container_info_list_init)
        return deployed

    def allocate_init(self, decision):
        migrations = self.allocation(decision)
        self.execution()
        return migrations

    def destroy_completed_containers(self):
        destroyed = []
        for i, container in enumerate(self.container_list):
            if container and container.get_base_ips() == 0:
                container.destroy()
                self.container_list[i] = None
                self.inactive_containers.append(container)
                destroyed.append(container)
        return destroyed

    def get_num_active_containers(self):
        num = 0
        for container in self.container_list:
            if container and container.active:
                num += 1
        return num

    def get_selectable_containers(self):
        selectable = []
        for container in self.container_list:
            if container and container.active and container.get_host_id() != -1:
                selectable.append(container.id)
        return selectable

    def add_containers(self, new_container_list):
        self.interval += 1
        destroyed = self.destroy_completed_containers()
        deployed = self.add_container_list(new_container_list)
        return deployed, destroyed

    def get_active_container_list(self):
        return [c.get_host_id() if c and c.active else -1 for c in self.container_list]

    def get_containers_in_hosts(self):
        return [len(self.get_containers_of_host(host)) for host in range(self.host_limit)]

    def simulation_step(self, decision, save_fitness=False):
        migrations = self.allocation(decision)
        if save_fitness:
            fitness = self.scheduler.allocation_fitness()
            self.execution()
            return migrations, fitness
        self.execution()
        return migrations

    def allocation(self, decision):
        router_bw_to_each = self.total_bw / len(decision) if len(decision) > 0 else self.total_bw
        migrations = list()
        for (cid, hid) in decision:
            container = self.get_container_by_id(cid)
            current_host_id = self.get_container_by_id(cid).get_host_id()
            current_host = self.get_host_by_id(current_host_id)
            target_host = self.get_host_by_id(hid)
            # todo PEP8 form in scheduler
            migrate_from_num = len(self.scheduler.get_migration_from_host(current_host_id, decision))
            migrate_to_num = len(self.scheduler.get_migration_to_host(hid, decision))

            download_bw = target_host.bw_cap.downlink / migrate_to_num
            upload_bw = current_host.bw_cap.uplink / migrate_from_num if current_host_id != -1 else np.inf
            alloc_bw = min(download_bw, upload_bw, router_bw_to_each)
            if hid != self.container_list[cid].host_id and self.get_placement_possible(cid, hid):
                migrations.append((cid, hid))
                container.allocate(hid, alloc_bw)

            # destroy unallocated container pointer in environment
        for i, container in enumerate(self.container_list):
            if container and container.get_host_id() == -1:
                self.container_list[i] = None
        return migrations

    def execution(self):
        for i, container in enumerate(self.container_list):
            if container:
                container.execute()
