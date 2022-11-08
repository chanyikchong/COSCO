from time import time, sleep
import multiprocessing
from joblib import Parallel, delayed

from framework.node.Node import Node
from framework.task.Task import Task
from framework.server.controller import RequestHandler

num_cores = multiprocessing.cpu_count()


class Framework:
    # Total power in watt
    # Total Router Bw
    # Interval Time in seconds
    def __init__(self, scheduler, container_limit, interval_time, host_init, database, env, logger):
        self.host_limit = len(host_init)
        self.scheduler = scheduler
        self.scheduler.set_environment(self)
        self.container_limit = container_limit
        self.host_list = list()
        self.container_list = list()
        self.interval_time = interval_time
        self.interval = 0
        self.db = database
        self.inactive_containers = list()
        self.logger = logger
        self.stats = None
        self.environment = env
        self.controller = RequestHandler(self.db, self)
        self.add_host_list_init(host_init)
        self.global_start_time = time()
        self.interval_alloc_timings = list()

    def add_host_init(self, ip, ips, ram, disk, bw, power_model):
        assert len(self.host_list) < self.host_limit
        host = Node(len(self.host_list), ip, ips, ram, disk, bw, power_model, self)
        self.host_list.append(host)

    def add_host_list_init(self, host_list):
        assert len(host_list) == self.host_limit
        for ip, ips, ram, disk, bw, power_model in host_list:
            self.add_host_init(ip, ips, ram, disk, bw, power_model)

    def add_container_init(self, creation_id, creation_interval, sla, application):
        container = Task(len(self.container_list), creation_id, creation_interval, sla, application, self, host_id=-1)
        self.container_list.append(container)
        return container

    def add_container_list_init(self, container_info_list):
        deployed = container_info_list[
                   :min(len(container_info_list), self.container_limit - self.get_num_active_containers())]
        deployed_containers = list()
        for creation_id, creation_interval, sla, application in deployed:
            dep = self.add_container_init(creation_id, creation_interval, sla, application)
            deployed_containers.append(dep)
        self.container_list += [None] * (self.container_limit - len(self.container_list))
        return [container.id for container in deployed_containers]

    def add_container(self, creation_id, creation_interval, sla, application):
        for i, c in enumerate(self.container_list):
            if c is None or not c.active:
                container = Task(i, creation_id, creation_interval, sla, application, self, host_id=-1)
                self.container_list[i] = container
                return container

    def add_container_list(self, container_info_list):
        deployed = container_info_list[
                   :min(len(container_info_list), self.container_limit - self.get_num_active_containers())]
        deployed_containers = list()
        for creation_id, creation_interval, sla, application in deployed:
            dep = self.add_container(creation_id, creation_interval, sla, application)
            deployed_containers.append(dep)
        return [container.id for container in deployed_containers]

    def get_containers_of_host(self, host_id):
        containers = list()
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
        ram_size_req, _, _ = container.get_ram()
        disk_size_req, _, _ = container.get_disk()
        ips_available = host.get_ips_available()
        ram_size_av, ram_read_av, ram_write_av = host.get_ram_available()
        disk_size_av, disk_read_av, disk_write_av = host.get_disk_available()
        return (ips_req <= ips_available and
                ram_size_req <= ram_size_av and
                disk_size_req <= disk_size_av)

    def add_containers_init(self, container_info_list_init):
        self.interval += 1
        deployed = self.add_container_list_init(container_info_list_init)
        return deployed

    def allocate_init(self, decision):
        start = time()
        migrations = list()
        for (cid, hid) in decision:
            container = self.get_container_by_id(cid)
            assert container.get_host_id() == -1 and hid != -1
            if self.get_placement_possible(cid, hid):
                migrations.append((cid, hid))
                container.allocate_and_execute(hid)
            # destroy pointer to this unallocated container as book-keeping is done by workload model
            else:
                self.container_list[cid] = None
        self.interval_alloc_timings.append(time() - start)
        self.logger.debug("First allocation: " + str(decision))
        self.logger.debug(
            'Interval allocation time for interval ' + str(self.interval) + ' is ' + str(self.interval_alloc_timings[-1]))
        print(
            'Interval allocation time for interval ' + str(self.interval) + ' is ' + str(self.interval_alloc_timings[-1]))
        self.visual_sleep(self.interval_time - self.interval_alloc_timings[-1])
        for host in self.host_list:
            host.update_utilization_metrics()
        return migrations

    def destroy_completed_containers(self):
        destroyed = []
        for i, container in enumerate(self.container_list):
            if container and not container.active:
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
        selectable = list()
        # selected = list()
        # containers = self.db.select("SELECT * FROM CreatedContainers;")
        for container in self.container_list:
            if container and container.active and container.get_host_id() != -1:
                selectable.append(container.id)
        print(selectable)
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

    def parallelized_func(self, i):
        cid, hid = i
        container = self.get_container_by_id(cid)
        if self.container_list[cid].host_id != -1:
            # raspi cannot migrate a container, uncomment this after this problem solved
            # container.allocate_and_restore(hid)
            container.allocate_and_execute(hid)
        else:
            container.allocate_and_execute(hid)
        return container

    def visual_sleep(self, t):
        total = str(int(t // 60)) + " min, " + str(t % 60) + " sec"
        for i in range(int(t)):
            print("\r>> Interval timer " + str(i // 60) + " min, " + str(i % 60) + " sec of " + total, end=' ')
            sleep(1)
        sleep(t % 1)
        print()

    def simulation_step(self, decision):
        start = time()
        migrations = list()
        container_ids_allocated = list()
        print(decision)
        for (cid, hid) in decision:
            container = self.get_container_by_id(cid)
            current_host_id = self.get_container_by_id(cid).get_host_id()
            current_host = self.get_host_by_id(current_host_id)
            target_host = self.get_host_by_id(hid)
            if hid != self.container_list[cid].host_id and self.get_placement_possible(cid, hid):
                container_ids_allocated.append(cid)
                migrations.append((cid, hid))
        Parallel(n_jobs=num_cores, backend='threading')(delayed(self.parallelized_func)(i) for i in migrations)
        for (cid, hid) in decision:
            if self.container_list[cid].host_id == -1:
                self.container_list[cid] = None
        self.interval_alloc_timings.append(time() - start)
        self.logger.debug("Decision: " + str(decision))
        self.logger.debug(
            'Interval allocation time for interval ' + str(self.interval) + ' is ' + str(self.interval_alloc_timings[-1]))
        print(
            'Interval allocation time for interval ' + str(self.interval) + ' is ' + str(self.interval_alloc_timings[-1]))
        self.visual_sleep(max(0, self.interval_time - self.interval_alloc_timings[-1]))
        for host in self.host_list:
            host.update_utilization_metrics()
        return migrations

    def __del__(self):
        for container in self.container_list:
            if container is not None:
                container.active = False
                container.destroy()
