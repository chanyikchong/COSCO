class Host:
    # IPS = Million Instructions per second capacity
    # RAM = Ram in MB capacity
    # Disk = Disk characteristics capacity
    # Bw = Bandwidth characteristics capacity
    def __init__(self, identity, ips, ram, disk, bw, latency, power_model, environment):
        self.id = identity
        self.ips_cap = ips
        self.ram_cap = ram
        self.disk_cap = disk
        self.bw_cap = bw
        self.latency = latency
        self.power_model = power_model
        self.power_model.alloc_host(self)
        self.power_model.host = self
        self.env = environment

    def get_power(self):
        return self.power_model.power()

    def get_power_max(self):
        return self.power_model.power_from_cpu(100)

    def get_power_from_ips(self, ips):
        return self.power_model.power_from_cpu(min(100, 100 * (ips / self.ips_cap)))

    def get_cpu(self):
        ips = self.get_apparent_ips()
        return 100 * (ips / self.ips_cap)

    def get_base_ips(self):
        # Get base ips count as sum of min ips of all containers
        ips = 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            ips += self.env.getContainerByID(containerID).get_base_ips()
        # assert ips <= self.ipsCap
        return ips

    def get_apparent_ips(self):
        # Give containers remaining IPS for faster execution
        ips = 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            ips += self.env.getContainerByID(containerID).get_apparent_ips()
        # assert int(ips) <= self.ipsCap
        return int(ips)

    def get_ips_available(self):
        # IPS available is ipsCap - baseIPS
        # When containers allocated, existing ips can be allocated to
        # the containers
        return self.ips_cap - self.get_base_ips()

    def get_current_ram(self):
        size, read, write = 0, 0, 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            s, r, w = self.env.getContainerByID(containerID).getRAM()
            size += s
            read += r
            write += w
        # assert size <= self.ramCap.size
        # assert read <= self.ramCap.read
        # assert write <= self.ramCap.write
        return size, read, write

    def get_ram_available(self):
        size, read, write = self.get_current_ram()
        return self.ram_cap.size - size, self.ram_cap.read - read, self.ram_cap.write - write

    def get_current_disk(self):
        size, read, write = 0, 0, 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            s, r, w = self.env.getContainerByID(containerID).getDisk()
            size += s
            read += r
            write += w
        assert size <= self.disk_cap.size
        assert read <= self.disk_cap.read
        assert write <= self.disk_cap.write
        return size, read, write

    def get_disk_available(self):
        size, read, write = self.get_current_disk()
        return self.disk_cap.size - size, self.disk_cap.read - read, self.disk_cap.write - write
