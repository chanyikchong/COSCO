from .Scheduler import Scheduler


class TMMTRScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        selected_host_ids = self.threshold_host_selection()
        selected_vm_ids = self.mmt_container_selection(selected_host_ids)
        return selected_vm_ids

    def placement(self, container_ids):
        return self.random_placement(container_ids)
