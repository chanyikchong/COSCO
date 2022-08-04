from .Scheduler import Scheduler


class RFScheduler(Scheduler):
    def __init__(self):
        super().__init__()

    def selection(self):
        return self.random_container_selection()

    def placement(self, container_ids):
        return self.first_fit_placement(container_ids)
