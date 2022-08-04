import math


class PM:
    def __init__(self):
        self.host = None
        self.power_list = list()

    def alloc_host(self, h):
        self.host = h

    # cpu consumption
    def power_from_cpu(self, cpu):
        index = math.floor(cpu / 10)
        left = self.power_list[index]
        right = self.power_list[index + 1 if cpu % 10 != 0 else index]
        alpha = (cpu / 10) - index
        return alpha * right + (1 - alpha) * left

    # cpu consumption in 100
    def power(self):
        cpu = self.host.get_cpu()
        index = math.floor(cpu / 10)
        left = self.power_list[index]
        right = self.power_list[index + 1 if cpu % 10 != 0 else index]
        alpha = (cpu / 10) - index
        return alpha * right + (1 - alpha) * left
