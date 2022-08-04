from .IPSM import IPSM


class IPSMBitbrain(IPSM):
    def __init__(self, ips_list, max_ips, duration, sla):
        super().__init__()
        self.ips_list = ips_list
        self.max_ips = max_ips
        self.sla = sla
        self.duration = duration
        self.completed_instructions = 0
        self.total_instructions = 0

    def get_ips(self):
        if self.total_instructions == 0:
            for ips in self.ips_list[:self.duration]:
                self.total_instructions += ips * self.container.env.intervaltime
        if self.completed_instructions < self.total_instructions:
            return self.ips_list[(self.container.env.interval - self.container.start_at) % len(self.ips_list)]
        return 0

    def get_max_ips(self):
        return self.max_ips
