from .Scheduler import Scheduler


class TMCRScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.utilHistoryContainer = []

    def update_util_history_container(self):
        container_util = [(cid.get_base_ips() if cid else 0) for cid in self.env.container_list]
        self.utilHistoryContainer.append(container_util)

    def selection(self):
        self.update_util_history_container()
        selected_host_ids = self.threshold_host_selection()
        selected_vm_ids = self.max_cor_container_selection(selected_host_ids, self.utilHistoryContainer)
        return selected_vm_ids

    def placement(self, container_ids):
        return self.random_placement(container_ids)
