from .Scheduler import Scheduler


class MADMCRScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.util_history = []
        self.util_history_container = []

    def update_util_history_container(self):
        container_util = [(cid.get_base_ips() if cid else 0) for cid in self.env.container_list]
        self.util_history_container.append(container_util)

    def update_util_history(self):
        host_utils = []
        for host in self.env.host_list:
            host_utils.append(host.get_cpu())
        self.util_history.append(host_utils)

    def selection(self):
        self.update_util_history_container()
        selected_host_ids = self.threshold_host_selection()
        selected_vm_ids = self.max_cor_container_selection(selected_host_ids, self.util_history_container)
        return selected_vm_ids

    def placement(self, container_ids):
        return self.random_placement(container_ids)
