# COSCO 笔记

# 主执行函数
## `main`
```python
datacenter, workload, scheduler, env, stats = initalizeEnvironment(env, logger)
```
初始化环境<br>
datacenter：数据中心<br>
workload：工作生成类<br>
scheduler：分配器算法<br>
env：环境<br>
stats：数据记录类<br>
并预先执行一步模拟过程
```python
for step in range(NUM_SIM_STEPS):
    print(color.BOLD + "Simulation Interval:", step, color.ENDC)
    stepSimulation(workload, scheduler, env, stats)
    if env != '' and step % 10 == 0:
        saveStats(stats, datacenter, workload, env, end=False)

saveStats(stats, datacenter, workload, env)
```
通过stepSimulation执行一步的模拟

## `initalizeEnvironment`
```python
def initalizeEnvironment(environment, logger)
    if environment != '':
        # Initialize the db
        db = Database(DB_NAME, DB_HOST, DB_PORT)
```
生成数据库，在COSCO内用InfluxDB
```python
    if environment != '':
        datacenter = Datacenter(HOSTS_IP, environment, 'Virtual')
    else:
        datacenter = AzureFog(HOSTS)

# Initialize workload
''' Can be SWSD, BWGD2, Azure2017Workload, Azure2019Workload // DFW, AIoTW '''
    if environment != '':
        workload = DFW(NEW_CONTAINERS, 1.5, db)
    else:
        workload = BWGD2(NEW_CONTAINERS, 1.5)
```
各个环境初始化
```python
    scheduler = GOBIScheduler(model_name)  # GOBIScheduler('energy_latency_'+str(HOSTS))
```
初始化分配器
```python
    hostlist = datacenter.generateHosts()
```
初始化Host节点，datacenter根据是否使用env决定，Simulator环境下为固定Host设置，Framework环境下根据实际Host参数设置
```python
    if environment != '':
        env = Framework(scheduler, CONTAINERS, INTERVAL_TIME, hostlist, db, environment, logger)
    else:
        env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, CONTAINERS, INTERVAL_TIME, hostlist)

    stats = Stats(env, workload, datacenter, scheduler)
```
初始化环境和数据记录类<br>

完成初始化后需要先执行一步模拟，包括生成containers和初始任务调度。
```python
    newcontainerinfos = workload.generateNewContainers(env.interval)
```
生成containers的信息tuple组成的list，`env.interval`是时间标签ID，用于记录container的生成时间区间。
```python
    deployed = env.addContainersInit(newcontainerinfos)
```
根据containers的信息tuple生成container对象并储存在`env.containerlist`中，deployed为需要schedule的container IDs。
```python
    start = time()
    decision = scheduler.placement(deployed)  # Decide placement using container ids
    schedulingTime = time() - start
```
scheduler根据deployed的ID list给出decision。decision的形式是(container_id, host_id)
```python
migrations = env.allocateInit(decision)
```
根据decision给出初始化的migration。
```python
    workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
```
更新workload里container的deploy状态
```python
    stats.saveStats(deployed, migrations, [], deployed, decision, schedulingTime)
```
保存当前interval的信息

## `stepSimulation`
```python
def stepSimulation(workload, scheduler, env, stats):
    newcontainerinfos = workload.generateNewContainers(env.interval)
    deployed, destroyed = env.addContainers(newcontainerinfos)
```
生成新的container对象，并将已完成的containerdestroy。
```python
    selected = scheduler.selection()  # Select container IDs for migration
    decision = scheduler.filter_placement(scheduler.placement(selected + deployed))
```
用scheduler选择需要migrate的container，为需要migrate和新生成deploy的container决定allocation。
```python
    migrations = env.simulationStep(decision) 
```

```python
    workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
    stats.saveStats(deployed, migrations, [], deployed, decision, schedulingTime)
```
更新workload里container的deploy状态，并保存当前interval的信息。

## `saveStats`
```python
def saveStats(stats, datacenter, workload, env, end=True):
    ... # 设置文件名，并创建文件
    stats.generateDatasets(dirname)
```
生成数据
```python
    if 'Datacenter' in datacenter.__class__.__name__:
        saved_env, saved_workload, saved_datacenter, saved_scheduler, saved_sim_scheduler = stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler
        stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler = None, None, None, None, None
        with open(dirname + '/' + dirname.split('/')[1] + '.pk', 'wb') as handle:
            pickle.dump(stats, handle)
        stats.env, stats.workload, stats.datacenter, stats.scheduler, stats.simulated_scheduler = saved_env, saved_workload, saved_datacenter, saved_scheduler, saved_sim_scheduler
```

```python
    if not end: 
        return
    stats.generateGraphs(dirname)
    stats.generateCompleteDatasets(dirname)
    stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
```
结束阶段，生成图片和数据集，重置stats中的参数。后续只保存stats数据不保存stats中的对象参数。
```python
    if 'Datacenter' in datacenter.__class__.__name__:
        stats.simulated_scheduler = None
        logger.getLogger().handlers.clear();
        env.logger.getLogger().handlers.clear()
        if os.path.exists(dirname + '/' + logFile): 
            os.remove(dirname + '/' + logFile)
        rename(logFile, dirname + '/' + logFile)
```

# 基类
## Workload
### `__init__`
```python
class Workload:
    def __init__(self):
        self.creation_id = 0
        self.createdContainers = []
        self.deployedContainers = []
```
creation_id记录container的ID，createdContainers记录所有生成的containers，deployedContainers记录对应ID的container是否已经deploy了，用bool表示。
### `getUndeployedContainers`
```python
    def getUndeployedContainers(self):
        undeployed = []
        for i, deployed in enumerate(self.deployedContainers):
            if not deployed:
                undeployed.append(self.createdContainers[i])
        return undeployed
```
获取没有被deploy的container，通过遍历`self.deployedContainers`获取
### `updateDeployedContainers`
```python
    def updateDeployedContainers(self, creationIDs):
        for cid in creationIDs:
            assert not self.deployedContainers[cid]
            self.deployedContainers[cid] = True
```
更新container的deploy状态

## Scheduler
### `__init__`

### `getMigrationToHost`
```python
    def getMigrationToHost(self, hostID, decision):
        containerIDs = []
        for (cid, hid) in decision:
            if hid == hostID:
                containerIDs.append(cid)
        return containerIDs
```
根据host_id和decision确定本次migration中有哪些container需要migration到当前host。

### `getMigrationFromHost`
```python
    def getMigrationFromHost(self, hostID, decision):
        containerIDs = []
        for (cid, _) in decision:
            hid = self.env.getContainerByID(cid).getHostID()
            if hid == hostID:
                containerIDs.append(cid)
        return containerIDs
```
获取需要从host ID转移的container ID。

## Stats
### `__init__`
```python
class Stats:
    def __init__(self, Environment, WorkloadModel, Datacenter, Scheduler):
        ... # 参数赋值
        self.simulated_scheduler = GOBIScheduler('energy_latency_' + str(self.datacenter.num_hosts))
        self.simulated_scheduler.env = self.env
        self.initStats()
```
包含一个模拟的scheduler。

### `initStats`
```python
    def initStats(self):
        self.hostinfo = []
        self.workloadinfo = []
        self.activecontainerinfo = []
        self.allcontainerinfo = []
        self.metrics = []
        self.schedulerinfo = []
```
各个保存信息的list

### `saveStats`
```python
    def saveStats(self, deployed, migrations, destroyed, selectedcontainers, decision, schedulingtime):
        self.saveHostInfo()
        self.saveWorkloadInfo(deployed, migrations)
        self.saveContainerInfo()
        self.saveAllContainerInfo()
        self.saveMetrics(destroyed, migrations)
        self.saveSchedulerInfo(selectedcontainers, decision, schedulingtime)
```
保存各个信息的接口函数。

### `saveHostInfo`
```python
    def saveHostInfo(self):
        hostinfo = dict()
        ... # 保存interval，cpu，每个host的container_num，power，ips，ram，disk
        self.hostinfo.append(hostinfo)
```
保存每一个host在一个interval的信息

### `saveWorkloadInfo`
```python
    def saveWorkloadInfo(self, deployed, migrations):
        workloadinfo = dict()
        workloadinfo['interval'] = self.env.interval
        workloadinfo['totalcontainers'] = len(self.workload.createdContainers)
        if self.workloadinfo != []:
            workloadinfo['newcontainers'] = workloadinfo['totalcontainers'] - self.workloadinfo[-1]['totalcontainers']
        else:
            workloadinfo['newcontainers'] = workloadinfo['totalcontainers']
        workloadinfo['deployed'] = len(deployed)
        workloadinfo['migrations'] = len(migrations)
        workloadinfo['inqueue'] = len(self.workload.getUndeployedContainers())
```
新增加的container通过相减获得。队列为undeploy container。

### `saveContainerInfo`
```python
    def saveContainerInfo(self):
        containerinfo = dict()
        ... # 保存
        self.activecontainerinfo.append(containerinfo)
```
保存active的container，从env.containerlist中获取。

### `saveAllContainerInfo`
```python
    def saveAllContainerInfo(self):
        containerinfo = dict()
        allCreatedContainers = [self.env.getContainerByCID(cid) for cid in list(np.where(self.workload.deployedContainers)[0])]
```
allCreatedContainers包含active和destroy的containers，不包含undeployed的containers。
```python
        if self.datacenter.__class__.__name__ == 'Datacenter':
            containerinfo['application'] = [self.env.getContainerByCID(cid).application for cid in list(np.where(self.workload.deployedContainers)[0])]
        ... # 保存信息
        self.allcontainerinfo.append(containerinfo)
```
对于Datacenter类额外保存application信息，

### `saveMetrics`
```python
    def saveMetrics(self, destroyed, migrations):
        metrics = dict()
        metrics['interval'] = self.env.interval
        metrics['numdestroyed'] = len(destroyed)
        metrics['nummigrations'] = len(migrations)
        metrics['energy'] = [host.getPower() * self.env.intervaltime for host in self.env.hostlist]  # 每个host在当前interval的消耗
        metrics['energytotalinterval'] = np.sum(metrics['energy'])  # 总消耗
        metrics['energypercontainerinterval'] = np.sum(metrics['energy']) / self.env.getNumActiveContainers()  # 当前interval每个active container的平均energy消耗。
        metrics['responsetime'] = [c.totalExecTime + c.totalMigrationTime for c in destroyed]  # 当前interval destroy的container的response time。
        metrics['avgresponsetime'] = np.average(metrics['responsetime']) if len(destroyed) > 0 else 0  # 当前interval的平均response time
        metrics['migrationtime'] = [c.totalMigrationTime for c in destroyed]  # 当前interval destroy的container的migration time
        metrics['avgmigrationtime'] = np.average(metrics['migrationtime']) if len(destroyed) > 0 else 0  # 平均migration time
        metrics['slaviolations'] = len(np.where([c.destroyAt > c.sla for c in destroyed]))  SLA violation的数目
        metrics['slaviolationspercentage'] = metrics['slaviolations'] * 100.0 / len(destroyed) if len(destroyed) > 0 else 0  # SLA violation比例
        metrics['waittime'] = [c.startAt - c.createAt for c in destroyed]  # 每个destroy container的等待时间
        metrics['energytotalinterval_pred'], metrics['avgresponsetime_pred'] = self.runSimulationGOBI()  # 用GOBI估计的interval的总energy和平均response time。
        self.metrics.append(metrics)
```
计算performance信息。

### `saveSchedulerInfo`
```python
    def saveSchedulerInfo(self, selectedcontainers, decision, schedulingtime):
        schedulerinfo = dict()
        schedulerinfo['interval'] = self.env.interval
        schedulerinfo['selection'] = selectedcontainers
        schedulerinfo['decision'] = decision  # scheduler 给出的decision
        schedulerinfo['schedule'] = [(c.id, c.getHostID()) if c else (None, None) for c in self.env.containerlist]  # 实际可执行的migration
        schedulerinfo['schedulingtime'] = schedulingtime
        if self.datacenter.__class__.__name__ == 'Datacenter':
            schedulerinfo['migrationTime'] = self.env.intervalAllocTimings[-1]
        self.schedulerinfo.append(schedulerinfo)
```

### `generateDatasetWithInterval`
```python
    def generateDatasetWithInterval(self, dirname, metric, objfunc, metric2=None, objfunc2=None):
        title = ''
        totalInterval = 
        metric_with_interval = []   # metric is of host
        metric2_with_interval = []  # metric2 is of containers
        host_alloc_with_interval = []
        objfunc2_with_interval = []
        objfunc_with_interval = []
        for interval in range(totalIntervals - 1):
            ...
            objfunc_with_interval.append(self.metrics[interval + 1][objfunc])  # destroy的container是上一阶段完成的所以当前interval的performance在下一阶段
            if metric2:
                metric2_with_interval.append(self.activecontainerinfo[interval][metric2])
            if objfunc2:
                objfunc2_with_interval.append(self.metrics[interval + 1][objfunc2])
        df = pd.DataFrame(metric_with_interval)  
        ... # 生成dataframe，顺序：metric1, metric2, container_allocation, obj_func1, obj_fun2
```

### `generateDatasetWithInterval2`
```python
    def generateDatasetWithInterval2(self, dirname, metric, metric2, metric3, metric4, objfunc, objfunc2):
```
功能逻辑同上。

### `generateGraphs`
```python
    def generateGraphs(self, dirname):
```
画一系列图的接口函数

### `generateGraphsWithInterval`
x轴是interval，y轴是host/container的数据。
```python
    def generateGraphsWithInterval(self, dirname, listinfo, obj, metric, metric2=None):
        fig, axes = plt.subplots(len(listinfo[0][metric]), 1, sharex=True, figsize=(4, 0.5 * len(listinfo[0][metric])))  # 每个host/container有一副图
        ...
        x = list(range(totalIntervals))
        ylimit = 0
        ylimit2 = 0
        for hostID in range(len(listinfo[0][metric])):
            metric_with_interval.append([listinfo[interval][metric][hostID] for interval in range(totalIntervals)])  # 每个host/container的信息
            ylimit = max(ylimit, max(metric_with_interval[-1]))
            if metric2:
                ...
```
根据obj决定host或container，metric和metric2是数据。
```python
        for hostID in range(len(listinfo[0][metric])):
            axes[hostID].set_ylim(0, max(ylimit, ylimit2))
            axes[hostID].plot(x, metric_with_interval[hostID])
            if metric2:
                axes[hostID].plot(x, metric2_with_interval[hostID])
            axes[hostID].set_ylabel(obj[0].capitalize() + " " + str(hostID))
            axes[hostID].grid(b=True, which='both', color='#eeeeee', linestyle='-')
        plt.tight_layout(pad=0)
        plt.savefig(dirname + '/' + title + '.pdf')
        plt.cla()
        plt.clf()
        plt.close('all')
```
画图。最后关闭fig减少内存消耗

### `generateMetricsWithInterval`
```python
    def generateMetricsWithInterval(self, dirname):
        fig, axes = plt.subplots(9, 1, sharex=True, figsize=(4, 5))
        x = list(range(len(self.metrics)))
        res = {}
        for i, metric in enumerate(['numdestroyed', 'nummigrations', 'energytotalinterval', 'avgresponsetime', \
                                    'avgmigrationtime', 'slaviolations', 'slaviolationspercentage', 'waittime',
                                    'energypercontainerinterval']):
            metric_with_interval = [self.metrics[i][metric] for i in range(len(self.metrics))] if metric != 'waittime' else [sum(self.metrics[i][metric]) for i in range(len(self.metrics))]
            ···
```
画总体performance的图

### `generateWorkloadWithInterval`
```python
    def generateWorkloadWithInterval(self, dirname):
        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(4, 5))
        x = list(range(len(self.workloadinfo)))
        for i, metric in enumerate(['totalcontainers', 'newcontainers', 'deployed', 'migrations', 'inqueue']):
            metric_with_interval = [self.workloadinfo[i][metric] for i in range(len(self.workloadinfo))]
```
画总体workload的图

### `generateCompleteDatasets`
```python
    def generateCompleteDatasets(self, dirname):
```
生成完整数据的接口

### `generateCompleteDataset`
```python
    def generateCompleteDataset(self, dirname, data, name):
        title = name + '_with_interval'
        metric_with_interval = []
        headers = list(data[0].keys())
        for datum in data:
            metric_with_interval.append([datum[value] for value in datum.keys()])
        df = pd.DataFrame(metric_with_interval, columns=headers)
        df.to_csv(dirname + '/' + title + '.csv', index=False)
```
保存特定数据list到dataframe。

# Simulator环境
该部分为使用Simulator环境下，各个类的执行说明
```python
datacenter = AzureFog(HOSTS)
workload = BWGD2(NEW_CONTAINERS, 1.5)
env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, CONTAINERS, INTERVAL_TIME, hostlist)
```
## Simulator类
### `__init__`
```python
class Simulator:
    # Total power in watt
    # Total Router Bw
    # Interval Time in seconds
    def __init__(self, TotalPower, RouterBw, Scheduler, ContainerLimit, IntervalTime, hostinit):
        ...
        self.scheduler.setEnvironment(self)
        self.addHostlistInit(hostinit)
```
包含一个模拟的scheduler。

### `getCreationIDs`
```python
    def getCreationIDs(self, migrations, containerIDs):
        creationIDs = []
        for decision in migrations:
            if decision[0] in containerIDs:
                creationIDs.append(self.containerlist[decision[0]].creationID)
        return creationIDs
```
获得migration中实际container的ID

### `getNumActiveContainers`
```python
    def getNumActiveContainers(self):
        num = 0
        for container in self.containerlist:
            if container and container.active: 
                num += 1
        return num
```
遍历containerlist里的container类，通过container类的active参数获得active的container数目。

### `getContainerByID`
```python
    def getContainerByID(self, containerID):
        return self.containerlist[containerID]
```
container的ID即为containerlist的位置，通过ID获取container对象

### `getPlacementPossible`
```python
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
```
获取container的base IPS，RAM信息，DISK信息。获取host的可用IPS，可用RAM，可用DISK。返回Boolean（container需要的是否超过host的）

### `addHostlistInit`
```python
    def addHostlistInit(self, hostList):
        assert len(hostList) == self.hostlimit
        for IPS, RAM, Disk, Bw, Latency, Powermodel in hostList:
            self.addHostInit(IPS, RAM, Disk, Bw, Latency, Powermodel)
```
根据host list生成对象。

### `addHostInit`
```python
    def addHostInit(self, IPS, RAM, Disk, Bw, Latency, Powermodel):
        assert len(self.hostlist) < self.hostlimit
        host = Host(len(self.hostlist), IPS, RAM, Disk, Bw, Latency, Powermodel, self)
        self.hostlist.append(host)
```
根据host的tuple信息生成host对象。

### `addContainersInit`
初始化步骤的增加container
```python
    def addContainersInit(self, containerInfoListInit):
        self.interval += 1
        deployed = self.addContainerListInit(containerInfoListInit)
        return deployed
```
入参为container信息tuple形成的list。interval ID增加一，返回deploy的container对象的ID list。

### `addContainerListInit`
```python
    def addContainerListInit(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit - self.getNumActiveContainers())]
```
入参为container信息tuple形成的list（新生成的containers）。选择前$min(len(list),\space container上限-active数)$个container deploy。
```python
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainerInit(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
            deployedContainers.append(dep)
```
运用container tuple的信息生成container对象并加入之`self.containerlist`。
```python
        self.containerlist += [None] * (self.containerlimit - len(self.containerlist))
        return [container.id for container in deployedContainers]
```
通过添加None把`self.containerlist`补齐至`self.containerlimit`的长度。最终返回deployed的container对象的ID list。


### `addContainerInit`
```python
    def addContainerInit(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        container = Container(len(self.containerlist), CreationID, CreationInterval, IPSModel, RAMModel, DiskModel,
                              self, HostID=-1)
        self.containerlist.append(container)
        return container
```
创建Container对象

### `allocateInit`
```python
    def allocateInit(self, decision):
        migrations = []
        routerBwToEach = self.totalbw / len(decision)
```
路由器可平均分配bw给decision里的每个container。
```python
        for (cid, hid) in decision:
            container = self.getContainerByID(cid)
            assert container.getHostID() == -1
```
当前container没有被分配过到任何host。
```python
            numberAllocToHost = len(self.scheduler.getMigrationToHost(hid, decision))
            allocbw = min(self.getHostByID(hid).bwCap.downlink / numberAllocToHost, routerBwToEach)
```
获取需要migrate到当前host的container数量。<br>
根据host的downlink bw和需要migrate的数量，获得该host可以平均分配的bw量。最后取$min$(host平均带宽, 路由器平均带宽)为最终允许带宽。
```python
            if self.getPlacementPossible(cid, hid):
                if container.getHostID() != hid:
                    migrations.append((cid, hid))
                container.allocateAndExecute(hid, allocbw)
            # destroy pointer to this unallocated container as book-keeping is done by workload model
            else:
                self.containerlist[cid] = None
        return migrations
```
若符合migration条件则执行，且container对象转移并执行，将decision加入migration的list。返回实际执行的decision即migration。

以上为初始化阶段需要模拟一步的函数

### `addContainers`
模拟阶段的增加container
```python
    self.interval += 1
        destroyed = self.destroyCompletedContainers()
        deployed = self.addContainerList(newContainerList)
        return deployed, destroyed
```
检查env中已完成的container并destroy，用新的container信息生成新的container对象。返回destroy的container对象和deploy的container ID。

### `addContainerList`
```python
    def addContainerList(self, containerInfoList):
        deployed = containerInfoList[:min(len(containerInfoList), self.containerlimit - self.getNumActiveContainers())]
        deployedContainers = []
        for CreationID, CreationInterval, IPSModel, RAMModel, DiskModel in deployed:
            dep = self.addContainer(CreationID, CreationInterval, IPSModel, RAMModel, DiskModel)
            deployedContainers.append(dep)
        return [container.id for container in deployedContainers]
```
生成container对象，并返回可以deploy的container ID。与`addContainerListInit`相同。

### `addContainer`
```python
    def addContainer(self, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel):
        for i, c in enumerate(self.containerlist):
            if c == None or not c.active:
                container = Container(i, CreationID, CreationInterval, IPSModel, RAMModel, DiskModel, self, HostID=-1)
                self.containerlist[i] = container
                return container
```
检查env.containerlist中是否有空位，若有则将新的container放入。与`addContainerInit`相似

### `destroyCompletedContainers`
```python
    def destroyCompletedContainers(self):
        destroyed = []
        for i, container in enumerate(self.containerlist):
            if container and container.getBaseIPS() == 0:
                container.destroy()
                self.containerlist[i] = None
                self.inactiveContainers.append(container)
                destroyed.append(container)
        return destroyed
```
若container无剩余IPS则需要destroy，重置env的containerlist，记录destroy的container。返回destroy的container

### `simulateionStep`
```python
    def simulationStep(self, decision):
        routerBwToEach = self.totalbw / len(decision) if len(decision) > 0 else self.totalbw
        migrations = []
        containerIDsAllocated = []
        for (cid, hid) in decision:
            container = self.getContainerByID(cid)
            currentHostID = self.getContainerByID(cid).getHostID()
            currentHost = self.getHostByID(currentHostID)
            targetHost = self.getHostByID(hid)
```
获取container，当前host，目标host对象
```python
            migrateFromNum = len(self.scheduler.getMigrationFromHost(currentHostID, decision))
            migrateToNum = len(self.scheduler.getMigrationToHost(hid, decision))
            allocbw = min(targetHost.bwCap.downlink / migrateToNum, currentHost.bwCap.uplink / migrateFromNum,
                          routerBwToEach)
```
根据当前host的migrate数量，目标host的migrate数量，计算带宽。取目标host平均值，当前host平均值，总路由带宽的平均值的最小值。
```python
            if hid != self.containerlist[cid].hostid and self.getPlacementPossible(cid, hid):
                migrations.append((cid, hid))
                container.allocateAndExecute(hid, allocbw)
                containerIDsAllocated.append(cid)
```
若该container需要migrate且可执行，则转移并执行，并记录其ID。
```python
        # destroy pointer to unallocated containers as book-keeping is done by workload model
        for (cid, hid) in decision:
            if self.containerlist[cid].hostid == -1: self.containerlist[cid] = None
```
若无法转移和安置成功则重置containerlist对应位置
```python
        for i, container in enumerate(self.containerlist):
            if container and i not in containerIDsAllocated:
                container.execute(0)
        return migrations
```
若container没有转移则直接执行container的execute（设置lastMigrationTime为0）。返回转移list

## AzureFog 类
```python
class AzureFog: 
    def __init__(self):
        ... 
```
### `generateHosts`
```python
    def generateHosts(self):
        hosts = []
        types = ['B2s', 'B2s', 'B2s', 'B2s', 'B4ms', 'B4ms', 'B4ms', 'B4ms', 'B8ms', 'B8ms'] * 5
        for i in range(self.num_hosts):
            typeID = types[i]
            IPS = self.types[typeID]['IPS']
            Ram = RAM(self.types[typeID]['RAMSize'], self.types[typeID]['RAMRead'] * 5,
                      self.types[typeID]['RAMWrite'] * 5)
            Disk_ = Disk(self.types[typeID]['DiskSize'], self.types[typeID]['DiskRead'] * 5,
                         self.types[typeID]['DiskWrite'] * 10)
            Bw = Bandwidth(self.types[typeID]['BwUp'], self.types[typeID]['BwDown'])
            Power = eval(self.types[typeID]['Power'] + '()')
            Latency = 0.003 if i < self.edge_hosts else 0.076
            hosts.append((IPS, Ram, Disk_, Bw, Latency, Power))
        return hosts
```
生成host的信息tuple，并没有生成host对象。在simulator的`addHostlistInit`中生成对象

## BitbrainWorkload2类
### `__init__`
```python
ips_multiplier = 2054.0 / (2 * 600)
class BWGD2(Workload):
    def __init__(self, meanNumContainers, sigmaNumContainers):
        super().__init__()
        self.mean = meanNumContainers
        self.sigma = sigmaNumContainers
        dataset_path = 'simulator/workload/datasets/bitbrain/'
        if not path.exists(dataset_path):
            # download data
```
继承`Workload`类，以container数目的期望和方差为入参，若无数据路径则会下载。
```python
        self.dataset_path = dataset_path
        self.disk_sizes = [1, 2, 3]
        self.meanSLA, self.sigmaSLA = 20, 3
        self.possible_indices = []
```
container硬盘数目list，SLA的均值与方差，`self.possible_indices`记录数据集中可以用作生成container的数据
```python
        for i in range(1, 500):
            df = pd.read_csv(self.dataset_path + 'rnd/' + str(i) + '.csv', sep=';\t')
            if (ips_multiplier * df['CPU usage [MHZ]']).to_list()[10] < 3000 and \
                    (ips_multiplier * df['CPU usage [MHZ]']).to_list()[10] > 500:
                self.possible_indices.append(i)
```
Bitbrain数据集中共有500台VM数据，每个VM有8344个时间数据。若index=10的时间里，`500 < CPU_usage*ips_multiplier < 3000`则为可用作生成的container

### `generateNewContainers`
```python
    def generateNewContainers(self, interval):
        workloadlist = []
        for i in range(max(1, int(gauss(self.mean, self.sigma)))):
```
interval，作为时间戳入参，记录container生成时间
```python
            CreationID = self.creation_id
            index = self.possible_indices[randint(0, len(self.possible_indices) - 1)]
            df = pd.read_csv(self.dataset_path + 'rnd/' + str(index) + '.csv', sep=';\t')
            sla = gauss(self.meanSLA, self.sigmaSLA)
```
用Uniform distribution选择VM，用Gaussian生成该container的SLA
```python
            IPSModel = IPSMBitbrain((ips_multiplier * df['CPU usage [MHZ]']).to_list(),
                                    (ips_multiplier * df['CPU capacity provisioned [MHZ]']).to_list()[0],
                                    int(1.2 * sla), interval + sla)
            RAMModel = RMBitbrain((df['Memory usage [KB]'] / 4000).to_list(),
                                  (df['Network received throughput [KB/s]'] / 1000).to_list(),
                                  (df['Network transmitted throughput [KB/s]'] / 1000).to_list())
            disk_size = self.disk_sizes[index % len(self.disk_sizes)]
            DiskModel = DMBitbrain(disk_size, (df['Disk read throughput [KB/s]'] / 4000).to_list(),
                                   (df['Disk write throughput [KB/s]'] / 12000).to_list())
            workloadlist.append((CreationID, interval, IPSModel, RAMModel, DiskModel))
            self.creation_id += 1
```
生成IPS model，RAM model，Disk model。container的disk size根据index决定。
```python
        self.createdContainers += workloadlist
        self.deployedContainers += [False] * len(workloadlist)
```
将生成的containers记录到`self.createdContainers`，`self.deployedContainers`设置为`False`。
```python
        return self.getUndeployedContainers()
```
获取所有还没有deploy的containers

## Host 类
### `__init__`
```python
class Host:
    # IPS = Million Instructions per second capacity
    # RAM = Ram in MB capacity
    # Disk = Disk characteristics capacity
    # Bw = Bandwidth characteristics capacity
    def __init__(self, ID, IPS, RAM, Disk, Bw, Latency, Powermodel, Environment):
        ...
        self.powermodel.host = self
```
参数赋值，powermodel的host赋值本对象

### `getPowerFromIPS`
```python
    def getPowerFromIPS(self, ips):
        return self.powermodel.powerFromCPU(min(100, 100 * (ips / self.ipsCap)))
```
根据IPS获得所需的power。

### `getCPU`
```python
    def getCPU(self):
        ips = self.getApparentIPS()
        return 100 * (ips / self.ipsCap)
```
计算执行apparentIPS需要多少CPU，百分比。

### `getBaseIPS`
```python
    def getBaseIPS(self):
        # Get base ips count as sum of min ips of all containers
        ips = 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            ips += self.env.getContainerByID(containerID).getBaseIPS()
        # assert ips <= self.ipsCap
        return ips
```
计算本host里所有container所需的最小IPS的和

### `getApparentIPS`
```python
    def getApparentIPS(self):
        # Give containers remaining IPS for faster execution
        ips = 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            ips += self.env.getContainerByID(containerID).getApparentIPS()
        # assert int(ips) <= self.ipsCap
        return int(ips)
```
计算本host里所有container可最快执行的IPS的和

### `getIPSAvailable`
```python
    def getIPSAvailable(self):
        return self.ipsCap - self.getBaseIPS()
```
可用IPS = 总IPS - container最小IPS

### RAM, DISK
```python
def getCurrentRAM(self):
        size, read, write = 0, 0, 0
        containers = self.env.getContainersOfHost(self.id)
        for containerID in containers:
            s, r, w = self.env.getContainerByID(containerID).getRAM()
            size += s
            read += r
            write += w
        return size, read, write
```
host中每个container所需资源的和
```python
    def getRAMAvailable(self):
        size, read, write = self.getCurrentRAM()
        return self.ramCap.size - size, self.ramCap.read - read, self.ramCap.write - write
```
容量减当前值。<br>
DISK相同。


## PowerModel (Host)
```python
class PM:
    def __init__(self):
        self.host = None

    def allocHost(self, h):
        self.host = h

    # cpu consumption in 100
    def powerFromCPU(self, cpu):
        index = math.floor(cpu / 10)
        left = self.powerlist[index]
        right = self.powerlist[index + 1 if cpu % 10 != 0 else index]
        alpha = (cpu / 10) - index
        return alpha * right + (1 - alpha) * left
```
根据CPU使用量确定power值，powerlist由特性power model继承获得（10个等级）。CPU值为0~100，最后结果用比例计算两个index占比。

```python
    def power(self):
        cpu = self.host.getCPU()
        index = math.floor(cpu / 10)
        left = self.powerlist[index]
        right = self.powerlist[index + 1 if cpu % 10 != 0 else index]
        alpha = (cpu / 10) - index
        return alpha * right + (1 - alpha) * left
```
同上但自带获取CPU， CPU值为apparentIPS的CPU



## Container 类
### `__init__`
```python
class Container:
    # IPS = ips requirement
    # RAM = ram requirement in MB
    # Size = container size in MB
    def __init__(self, ID, creationID, creationInterval, IPSModel, RAMModel, DiskModel, Environment, HostID=-1):
```
创建对象，ID为env.containerlist中ID，creationID为workload中的创建ID
### `getBaseIPS`
```python
    def getBaseIPS(self):
        return self.ipsmodel.getIPS()
```
获得剩余instruction需要的最小IPS

### `getApparentIPS`
获取单位时间最多可使用的IPS
```python
    def getApparentIPS(self):
        if self.hostid == -1: 
            return self.ipsmodel.getMaxIPS()
        hostBaseIPS = self.getHost().getBaseIPS()
        hostIPSCap = self.getHost().ipsCap
        canUseIPS = (hostIPSCap - hostBaseIPS) / len(self.env.getContainersOfHost(self.hostid))
```
可以使用的IPS=hostIPSCap - hostBaseIPS平均分配至当前host所要处理的container数目
```python
        if canUseIPS < 0:
            return 0
        return min(self.ipsmodel.getMaxIPS(), self.getBaseIPS() + canUseIPS)
```
取$min$(container IPS最大值，container IPS base值+host可用值)为模拟执行的IPS值

### ```getRAM```
```python
    def getRAM(self):
        rsize, rread, rwrite = self.rammodel.ram()
        self.lastContainerSize = rsize
        return rsize, rread, rwrite
```
通过RAM model获取RAM信息，container的初始size是RAM的size。

### `getContainerSize`
```python
    def getContainerSize(self):
        if self.lastContainerSize == 0: 
            self.getRAM()
        return self.lastContainerSize
```
获取container的size

### `allocate`
```python
    def allocate(self, hostID, allocBw):
        lastMigrationTime = 0
        if self.hostid != hostID:
            lastMigrationTime += self.getContainerSize() / allocBw
```
migration_time增加，container size / 带宽 为migration时间。
```python
            lastMigrationTime += abs(self.env.hostlist[self.hostid].latency - self.env.hostlist[hostID].latency)
        self.hostid = hostID
        return lastMigrationTime
```
原host的latency - 转移host的latency 为增加的migration time。最后更改container的host。

### `execute`
```python
    def execute(self, lastMigrationTime):
        assert self.hostid != -1
        self.totalMigrationTime += lastMigrationTime
        execTime = self.env.intervaltime - lastMigrationTime
```
叠加总migration时间<br>
本段interval剩余执行时间=interval时间-migration时间。
```python
        apparentIPS = self.getApparentIPS()
        requiredExecTime = (self.ipsmodel.totalInstructions - self.ipsmodel.completedInstructions) / apparentIPS if apparentIPS else 0
        self.totalExecTime += min(execTime, requiredExecTime)
```
还需要的执行时间=total-completed/可使用IPS。本interval共可执行时间$min$(剩余执行时间，需要的执行时间)。
```python
        self.ipsmodel.completedInstructions += apparentIPS * min(execTime, requiredExecTime)
```
已完成的instruction计算。

### `allocateAndExecute`
```python
    def allocateAndExecute(self, hostID, allocBw):
        self.execute(self.allocate(hostID, allocBw))
```

### `destroy`
```python
    def destroy(self):
        self.destroyAt = self.env.interval
        self.hostid = -1
        self.active = False
```
container完成，记录完成时间，重置host和active

## IPSModel (Container)
### `getIPS`
```python
    def getIPS(self):
        if self.totalInstructions == 0:
            for ips in self.ips_list[:self.duration]:
                self.totalInstructions += ips * self.container.env.intervaltime
        if self.completedInstructions < self.totalInstructions:
            return self.ips_list[(self.container.env.interval - self.container.startAt) % len(self.ips_list)]
        return 0
```
计算总instruction，获取bitbrain里对应的ips并乘以interval时长。<br>
获取当前的IPS，对应时间端bitbrain的IPS

## DiskModel (container)
### `disk`
```python
    def disk(self):
        read_list_count = (self.container.env.interval - self.container.startAt) % len(self.read_list)
        write_list_count = (self.container.env.interval - self.container.startAt) % len(self.write_list)
        return self.constant_size, self.read_list[read_list_count], self.write_list[write_list_count]
```
获取read，write的时间，在list中取对应数据

## RAMModel (container)
### `ram`
```python
    def ram(self):
        size_list_count = (self.container.env.interval - self.container.startAt) % len(self.size_list)
        read_list_count = (self.container.env.interval - self.container.startAt) % len(self.read_list)
        write_list_count = (self.container.env.interval - self.container.startAt) % len(self.write_list)
        return self.size_list[size_list_count], self.read_list[read_list_count], self.write_list[write_list_count]
```
同Disk model。


# Framework环境
