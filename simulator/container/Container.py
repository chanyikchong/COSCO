class Container:
    # IPS = ips requirement
    # RAM = ram requirement in MB
    # Size = container size in MB
    def __init__(self, identity, creation_id, creation_interval, ips_model, ram_model, disk_model, environment,
                 host_id=-1):
        self.id = identity
        self.creation_id = creation_id
        self.ips_model = ips_model
        self.ips_model.alloc_container(self)
        self.sla = self.ips_model.sla
        self.ram_model = ram_model
        self.ram_model.alloc_container(self)
        self.disk_model = disk_model
        self.disk_model.alloc_container(self)
        self.host_id = host_id
        self.env = environment
        self.create_at = creation_interval
        self.start_at = self.env.interval
        self.total_exec_time = 0
        self.total_migration_time = 0
        self.active = True
        self.destroy_at = -1
        self.last_container_size = 0
        self.last_migration_time = 0

    def get_base_ips(self):
        return self.ips_model.get_ips()

    def get_apparent_ips(self):
        if self.host_id == -1:
            return self.ips_model.get_max_ips()
        host_base_ips = self.get_host().get_base_ips()
        host_ips_cap = self.get_host().ips_cap
        can_use_ips = (host_ips_cap - host_base_ips) / len(self.env.get_containers_of_host(self.host_id))
        if can_use_ips < 0:
            return 0
        return min(self.ips_model.get_max_ips(), self.get_base_ips() + can_use_ips)

    def get_ram(self):
        r_size, r_read, r_write = self.ram_model.ram()
        self.last_container_size = r_size
        return r_size, r_read, r_write

    def get_disk(self):
        return self.disk_model.disk()

    def get_container_size(self):
        if self.last_container_size == 0:
            self.get_ram()
        return self.last_container_size

    def get_host_id(self):
        return self.host_id

    def get_host(self):
        return self.env.get_host_by_id(self.host_id)

    def allocate(self, host_id, alloc_bw):
        # Migrate if allocated to a different host
        # Migration time is sum of network latency
        # and time to transfer container based on
        # network bandwidth and container size.
        if self.host_id != host_id:
            self.last_migration_time += self.get_container_size() / alloc_bw
            self.last_migration_time += abs(
                self.env.host_list[self.host_id].latency - self.env.host_list[host_id].latency)
        self.host_id = host_id
        return self.last_migration_time

    def execute(self):
        # Migration time is the time to migrate to new host
        # Thus, execution of task takes place for interval
        # time - migration time with apparent ips
        assert self.host_id != -1
        self.total_migration_time += self.last_migration_time
        exec_time = self.env.interval_time - self.last_migration_time
        apparent_ips = self.get_apparent_ips()
        remain_instructions = self.ips_model.total_instructions - self.ips_model.completed_instructions
        required_exec_time = remain_instructions / apparent_ips if apparent_ips else 0
        self.total_exec_time += min(exec_time, required_exec_time)
        self.ips_model.completed_instructions += apparent_ips * min(exec_time, required_exec_time)
        self.last_migration_time = 0

    def allocate_execute(self, host_id, alloc_bw):
        self.allocate(host_id, alloc_bw)
        self.execute()

    def destroy(self):
        self.destroy_at = self.env.interval
        self.host_id = -1
        self.active = False
