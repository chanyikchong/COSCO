from .IPSM import IPSM


class IPSMConstant(IPSM):
    def __init__(self, constant_ips, max_ips, duration, sla):
        super().__init__()
        self.constant_ips = constant_ips
        self.max_ips = max_ips
        self.sla = sla
        self.duration = duration
        self.completed_instructions = 0
        self.total_instructions = 0

    def get_ips(self):
        self.total_instructions = self.constant_ips * self.duration * self.container.env.interval_time
        if self.completed_instructions < self.total_instructions:
            return self.constant_ips
        return 0

    def get_max_ips(self):
        return self.max_ips
