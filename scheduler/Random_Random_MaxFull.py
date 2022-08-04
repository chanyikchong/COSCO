from .Scheduler import Scheduler


class RMScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        return self.random_container_selection()

    def placement(self, container_ids):
        return self.max_full_placement(container_ids)
