from .PM import PM

# The power model of an IBM System x3650 M2 (Intel Xeon X5570 8 core, 2 chips, 4 cores/chip)
# The data is accessed from https://www.spec.org/power_ssj2008/results/res2014q4/power_ssj2008-20141023-00677.html

class PMXeon_X5570(PM):
    def __init__(self):
        super().__init__()
        self.power_list = [81.4, 110, 125, 139, 153, 167, 182, 199, 214, 229, 244]
