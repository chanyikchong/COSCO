from .Scheduler import Scheduler


class LRMMTRScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.util_history = []

    def update_util_history(self):
        host_utils = []
        for host in self.env.host_list:
            host_utils.append(host.get_cpu())
        self.util_history.append(host_utils)

    def selection(self):
        self.update_util_history()
        selected_host_ids = self.lr_selection(self.util_history)
        selected_vm_ids = self.mmt_container_selection(selected_host_ids)
        return selected_vm_ids

    def placement(self, container_ids):
        return self.random_placement(container_ids)
