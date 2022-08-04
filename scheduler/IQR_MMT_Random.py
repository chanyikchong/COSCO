from .Scheduler import Scheduler


class IQRMMTRScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.utilHistory = []

    def update_util_history(self):
        hostUtils = []
        for host in self.env.host_list:
            hostUtils.append(host.get_cpu())
        self.utilHistory.append(hostUtils)

    def selection(self):
        self.update_util_history()
        selected_host_ids = self.iqr_selection(self.utilHistory)
        selected_vm_ids = self.mmt_container_selection(selected_host_ids)
        return selected_vm_ids

    def placement(self, container_ids):
        return self.random_placement(container_ids)
