from .PM import PM


class PMConstant(PM):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant
        self.power_list = [constant] * 11

    # CPU consumption in 100
    def power(self):
        return self.constant
