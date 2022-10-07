from simulator.Simulator import Simulator
from my_package.my_host import MyHost


class MySimulator(Simulator):
    def __init__(self, total_power, router_bw, scheduler, container_limit, interval_time, host_init):
        super(MySimulator, self).__init__(total_power, router_bw, scheduler, container_limit, interval_time, host_init)

    def set_interval_time(self, t):
        self.interval_time = t

    def finish_execution(self, stats):
        max_t = [container.remain_exec_time() + container.last_migration_time for container in self.container_list if
                 container]
        max_t = max(max_t) if max_t else 0
        stats.save_power_info(self.host_list, max_t)
        self.set_interval_time(max_t)
        self.execution()

    def add_host_init(self, ips, ram, disk, bw, latency, power_model):
        assert len(self.host_list) < self.host_limit
        host = MyHost(len(self.host_list), ips, ram, disk, bw, latency, power_model, self)
        self.host_list.append(host)
