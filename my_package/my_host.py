from simulator.host import Host


class MyHost(Host):
    def __init__(self, identity, ips, ram, disk, bw, latency, power_model, environment):
        super(MyHost, self).__init__(identity, ips, ram, disk, bw, latency, power_model, environment)
        self.final_power = 0

    def get_final_power(self, execution_time):
        container_list = self.env.get_containers_of_host(self.id)
        print('in')
