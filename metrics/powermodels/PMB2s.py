from .PM import PM

# The power model of an IBM Corporation IBM System x3200 M2 (Intel Xeon E502673 v4 2 core, 1 chips, 2 cores/chip)
# The data is accessed from https://www.spec.org/power_ssj2008/results/res2011q1/power_ssj2008-20110124-00340.html

class PMB2s(PM):
    def __init__(self):
        super().__init__()
        self.power_list = [75.2, 78.2, 84.1, 89.6, 94.9, 100.0, 105.0, 109.0, 112.0, 115.0, 117.0]
