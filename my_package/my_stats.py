class MyStats:
    def __init__(self):
        self.host_input_info = list()
        self.container_input_info = list()
        self.old_allocation = list()
        self.new_allocation = list()
        self.power_info = list()
        self.response_time_info = list()

    def save_container_info(self, container_list):
        pass

    def save_host_info(self, host_list):
        pass

    def save_old_allocation(self, container_list):
        pass

    def save_new_allocation(self, container_list):
        pass

    def save_power_info(self, host_list, execution_time):
        for host in host_list:
            host.get_final_power(execution_time)
        pass
