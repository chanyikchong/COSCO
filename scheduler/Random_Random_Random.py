from .Scheduler import Scheduler


class RandomScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        return self.random_container_selection()

    def placement(self, container_ids):
        return self.random_placement(container_ids)
