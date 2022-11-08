from dateutil import parser
from datetime import datetime

from metrics.Disk import Disk
from metrics.RAM import RAM
from metrics.Bandwidth import Bandwidth


class Task:
    """
    Task work as a container
    # IPS = ips requirement
    # RAM = ram requirement in MB
    # Size = container size in MB
    """
    def __init__(self, id, creation_id, creation_interval, sla, application, framework, host_id=-1):
        self.id = id
        self.creation_id = creation_id
        # Initial utilization metrics
        self.ips = 0
        self.ram = RAM(0, 0, 0)
        self.bw = Bandwidth(0, 0)
        self.disk = Disk(0, 0, 0)
        self.sla = sla
        self.host_id = host_id
        self.json_body = dict()
        self.env = framework
        self.create_at = creation_interval
        self.start_at = self.env.interval
        self.last_read_bytes = 0
        self.last_write_bytes = 0
        self.total_exec_time = 0
        self.total_migration_time = 0
        self.active = True
        self.destroy_at = -1
        self.application = application
        self.exec_error = ""
        self.container_db_insert()

    def container_db_insert(self):
        # write container information to database
        self.json_body = {
            "measurement": "CreatedContainers",
            "tags": {
                "container_id": self.id,
                "container_creation_id": self.creation_id,

            },
            "creation_interval": self.create_at,
            "start_interval": self.start_at,
            "fields":
                {
                    "Host_id": self.host_id,
                    "name": str(self.creation_id) + "_" + str(self.id),
                    "image": self.application,
                    "Active": self.active,
                    "totalExecTime": self.total_exec_time,
                    "startAt": self.start_at,
                    "createAt": self.create_at,
                    "destroyAt": self.destroy_at,
                    "IPS": self.ips,
                    "SLA": self.sla,
                    "RAM_size": self.ram.size,
                    "RAM_read": self.ram.read,
                    "RAM_write": self.ram.write,
                    "DISK_size": self.disk.size,
                    "DISK_read": self.disk.read,
                    "DISK_write": self.disk.write,
                }
        }
        self.env.db.insert([self.json_body])

    def get_base_ips(self):
        return self.ips

    def get_apparent_ips(self):
        return self.ips

    def get_ram(self):
        return self.ram.size, self.ram.read, self.ram.write

    def get_disk(self):
        return self.disk.size, self.disk.read, self.disk.write

    def get_container_size(self):
        return self.ram.size

    def get_host_id(self):
        return self.host_id

    def get_host(self):
        return self.env.get_host_by_id(self.host_id)

    def allocate_and_execute(self, host_id):
        # self.env.logger.debug("Allocating container "+self.json_body['fields']['name']+" to host "+self.env.getHostByID(hostID).ip)
        self.host_id = host_id
        self.json_body["fields"]["Host_id"] = host_id
        _, last_migration_time = self.env.controller.create(self.json_body, self.env.get_host_by_id(self.host_id).ip)
        self.total_migration_time += last_migration_time
        exec_time = self.env.interval_time - last_migration_time
        self.total_exec_time += exec_time

    def allocate_and_restore(self, host_id):
        # self.env.logger.debug("Migrating container "+self.json_body['fields']['name']+" from host "+self.getHost().ip+
        # 	" to host "+self.env.getHostByID(hostID).ip)
        cur_host_ip = self.get_host().ip
        self.host_id = host_id
        tar_host_ip = self.get_host().ip
        self.json_body["fields"]["Host_id"] = host_id
        _, checkpoint_time = self.env.controller.checkpoint(self.creation_id, self.id, cur_host_ip)
        _, migration_time = self.env.controller.migrate(self.creation_id, self.id, cur_host_ip, tar_host_ip)
        _, restore_time = self.env.controller.restore(self.creation_id, self.id, self.application, tar_host_ip)
        last_migration_time = checkpoint_time + migration_time + restore_time
        self.total_migration_time += last_migration_time
        exec_time = self.env.interval_time - last_migration_time
        self.total_exec_time += exec_time

    def destroy(self):
        assert not self.active
        rc = self.env.controller.destroy(self.json_body, self.get_host().ip)
        # query = "DELETE FROM CreatedContainers WHERE creation_id="+"'"+str(self.creationID)+"'"+";"
        # self.env.db.delete_measurement(query)
        self.json_body["tags"]["active"] = False
        self.json_body["fields"]["Host_id"] = -1
        self.destroy_at = self.env.interval
        self.host_id = -1

    def update_utilization_metrics(self, data):
        self.ips = data['cpu'] * self.get_host().ips_cap / 100
        self.ram.size = data['memory'] * self.get_host().ram_cap.size / 100
        if self.last_read_bytes != 0:
            self.ram.read = (data['read_bytes'] - self.last_read_bytes) / (1024 * 1024 * self.env.interval_time)
            self.ram.write = (data['write_bytes'] - self.last_write_bytes) / (1024 * 1024 * self.env.interval_time)
            self.disk.read = (data['read_bytes'] - self.last_read_bytes) / (1024 * 1024 * self.env.interval_time)
            self.disk.write = (data['write_bytes'] - self.last_write_bytes) / (1024 * 1024 * self.env.interval_time)
        self.last_read_bytes = data['read_bytes']
        self.last_write_bytes = data['write_bytes']
        self.disk.size = float(data['disk'][:-1]) if data['disk'][-1] == 'M' else 1024 * float(data['disk'][:-1])
        self.bw.downlink = data['bw_down']
        self.bw.uplink = data['bw_up']
        self.active = data['running']
        if not self.active:
            finished_at = parser.parse(data['finished_at']).replace(tzinfo=None)
            now = datetime.utcnow()
            self.total_exec_time -= abs((now - finished_at).total_seconds())
            self.exec_error = data['error']
