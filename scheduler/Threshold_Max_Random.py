from .Scheduler import Scheduler


class TMRScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        selected_host_ids = self.threshold_host_selection()
        selected_ids = self.max_use_container_selection(selected_host_ids)
        return selected_ids

    def placement(self, container_ids):
        return self.random_placement(container_ids)
