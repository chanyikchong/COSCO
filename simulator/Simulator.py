import numpy as np

from simulator.host.Host import *
from simulator.container.Container import *


class Simulator:
    # Total power in watt
    # Total Router Bw
    # Interval Time in seconds
    def __init__(self, TotalPower, RouterBw, Scheduler, ContainerLimit, IntervalTime, hostinit):
        self.totalpower = TotalPower
        self.totalbw = RouterBw
        self.hostlimit = len(hostinit)
        self.scheduler = Scheduler
        self.scheduler.setEnvironment(self)
        self.containerlimit = ContainerLimit
        self.hostlist = []
        self.containerlist = []
        self.intervaltime = IntervalTime
        self.interval = 0
        self.inactiveContainers = []
        self.stats = None
        self.addHostlistInit(hostinit)

    def addHostInit(self, IPS, RAM, Disk, Bw, Latency, Powermodel):
        assert len(self.hostlist) < self.hostlimit
        host = Host(len(self.hostlist), IPS, RAM, Disk, Bw, Latency, Powermodel, self)
        self.hostlist.append(host)

    def addHostlistInit(self, hostList):
        assert len(hostList) == self.hostlimit
        for IPS, RAM, Disk, Bw, Latency, Powermodel in hostList:
            self.addHostInit(IPS, RAM, Disk, Bw, Latency, Powermodel)

    def addContainerInit(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        container = Container(len(self.containerlist), CreationID, CreationInterval, IPSModel, RAMModel, DiskModel,
                              self, HostID=-1)
        self.containerlist.append(container)
        return container

    def addContainerListInit(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit - self.getNumActiveContainers())]
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
            deployedContainers.append(dep)
        self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
        return [container.id for container in deployedContainers]

    def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        for i, c in enumerate(self.containerlist):
            if c == None or not c.active:
                container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID=-1)
                self.containerlist[i] = container
                return container

    def addContainerList(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit - self.getNumActiveContainers())]
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainer(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
            deployedContainers.append(dep)
        return [container.id for container in deployedContainers]

    def getContainersOfHost(self, hostID):
        containers = []
        for container in self.containerlist:
            if container and container.hostid == hostID:
                containers.append(container.id)
        return containers

    def getContainerByID(self, containerID):
        return self.containerlist[containerID]

    def getContainerByCID(self, creationID):
        for c in self.containerlist + self.inactiveContainers:
            if c and c.creationID == creationID:
                return c

    def getHostByID(self, hostID):
        # todo return last host when hostID = -1 BUG
        return self.hostlist[hostID]

    def getCreationIDs(self, migrations, containerIDs):
        creationIDs = []
        for decision in migrations:
            if decision[0] in containerIDs:
                creationIDs.append(self.containerlist[decision[0]].creationID)
        return creationIDs

    def getPlacementPossible(self, containerID, hostID):
        container = self.containerlist[containerID]
        host = self.hostlist[hostID]
        ipsreq = container.getBaseIPS()
        ramsizereq, ramreadreq, ramwritereq = container.getRAM()
        disksizereq, diskreadreq, diskwritereq = container.getDisk()
        ipsavailable = host.getIPSAvailable()
        ramsizeav, ramreadav, ramwriteav = host.getRAMAvailable()
        disksizeav, diskreadav, diskwriteav = host.getDiskAvailable()
        return (ipsreq <= ipsavailable and \
                ramsizereq <= ramsizeav and \
                # ramreadreq <= ramreadav and \
                # ramwritereq <= ramwriteav and \
                disksizereq <= disksizeav \
                # diskreadreq <= diskreadav and \
                # diskwritereq <= diskwriteav
                )

    def addContainersInit(self, containerInfoListInit):
        self.interval += 1
        deployed = self.addContainerListInit(containerInfoListInit)
        return deployed

    def allocateInit(self, decision):
        migrations = self.allocation(decision)
        self.execution()
        return migrations

    def destroyCompletedContainers(self):
        destroyed = []
        for i, container in enumerate(self.containerlist):
            if container and container.getBaseIPS() == 0:
                container.destroy()
                self.containerlist[i] = None
                self.inactiveContainers.append(container)
                destroyed.append(container)
        return destroyed

    def getNumActiveContainers(self):
        num = 0
        for container in self.containerlist:
            if container and container.active:
                num += 1
        return num

    def getSelectableContainers(self):
        selectable = []
        for container in self.containerlist:
            if container and container.active and container.getHostID() != -1:
                selectable.append(container.id)
        return selectable

    def addContainers(self, newContainerList):
        self.interval += 1
        destroyed = self.destroyCompletedContainers()
        deployed = self.addContainerList(newContainerList)
        return deployed, destroyed

    def getActiveContainerList(self):
        return [c.getHostID() if c and c.active else -1 for c in self.containerlist]

    def getContainersInHosts(self):
        return [len(self.getContainersOfHost(host)) for host in range(self.hostlimit)]

    def simulationStep(self, decision, save_fitness=False):
        migrations = self.allocation(decision)
        if save_fitness:
            fitness = self.scheduler.allocation_fitness()
            self.execution()
            return migrations, fitness
        self.execution()
        return migrations

    def allocation(self, decision):
        routerBwToEach = self.totalbw / len(decision) if len(decision) > 0 else self.totalbw
        migrations = []
        for (cid, hid) in decision:
            container = self.getContainerByID(cid)
            currentHostID = self.getContainerByID(cid).getHostID()
            currentHost = self.getHostByID(currentHostID)
            targetHost = self.getHostByID(hid)
            migrateFromNum = len(self.scheduler.getMigrationFromHost(currentHostID, decision))
            migrateToNum = len(self.scheduler.getMigrationToHost(hid, decision))
            download_bw = targetHost.bwCap.downlink / migrateToNum
            upload_bw = currentHost.bwCap.uplink / migrateFromNum if currentHostID != -1 else np.inf
            allocbw = min(download_bw, upload_bw, routerBwToEach)
            if hid != self.containerlist[cid].hostid and self.getPlacementPossible(cid, hid):
                migrations.append((cid, hid))
                container.allocate(hid, allocbw)

            # destroy unallocated container pointer in environment
        for i, container in enumerate(self.containerlist):
            if container and container.getHostID() == -1:
                self.containerlist[i] = None
        return migrations

    def execution(self):
        for i, container in enumerate(self.containerlist):
            if container:
                container.execute()
