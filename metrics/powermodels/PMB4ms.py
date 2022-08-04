from .PM import PM


# The power model of an IBM Corporation IBM System x3350 (Intel Xeon E502673 v3 4 core, 1 chips, 4 cores/chip)
# The data is accessed from https://www.spec.org/power_ssj2008/results/res2008q2/power_ssj2008-20080506-00052.html

class PMB4ms(PM):
    def __init__(self):
        super().__init__()
        self.power_list = [71.0, 77.9, 83.4, 89.2, 95.6, 102.0, 108.0, 114.0, 119.0, 123.0, 126.0]
