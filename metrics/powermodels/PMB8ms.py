from .PM import PM


# The power model of an Huawei Technologies Co., Ltd RH2288H V2 (Intel Xeon E502673 v3 8 core, 1 chips, 8 cores/chip)
# The data is accessed from https://www.spec.org/power_ssj2008/results/res2014q2/power_ssj2008-20140408-00655.html

class PMB8ms(PM):
    def __init__(self):
        super().__init__()
        self.power_list = [68.7, 78.3, 84.0, 88.4, 92.5, 97.3, 104.0, 111.0, 121.0, 131.0, 137.0]
