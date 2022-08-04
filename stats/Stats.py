import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scheduler.GOBI import GOBIScheduler

plt.style.use(['science'])
plt.rcParams["text.usetex"] = False


class Stats:
    def __init__(self, environment, workload_model, datacenter, scheduler):
        self.env = environment
        self.env.stats = self
        self.workload = workload_model
        self.datacenter = datacenter
        self.scheduler = scheduler
        self.simulated_scheduler = GOBIScheduler('energy_latency_' + str(self.datacenter.num_hosts))
        self.simulated_scheduler.env = self.env
        self.host_info = []
        self.workload_info = []
        self.active_container_info = []
        self.all_container_info = []
        self.metrics = []
        self.scheduler_info = []
        self._max_response_time = -np.inf
        self._temp_get_normalization()

    def save_host_info(self):
        host_info = dict()
        host_info['interval'] = self.env.interval
        host_info['cpu'] = [host.getCPU() for host in self.env.hostlist]
        host_info['num_containers'] = [len(self.env.getContainersOfHost(i)) for i, host in enumerate(self.env.hostlist)]
        host_info['power'] = [host.getPower() for host in self.env.hostlist]
        host_info['base_ips'] = [host.getBaseIPS() for host in self.env.hostlist]
        host_info['ips_available'] = [host.getIPSAvailable() for host in self.env.hostlist]
        host_info['ips_cap'] = [host.ipsCap for host in self.env.hostlist]
        host_info['apparent_ips'] = [host.getApparentIPS() for host in self.env.hostlist]
        host_info['ram'] = [host.getCurrentRAM() for host in self.env.hostlist]
        host_info['ram_available'] = [host.getRAMAvailable() for host in self.env.hostlist]
        host_info['disk'] = [host.getCurrentDisk() for host in self.env.hostlist]
        host_info['disk_available'] = [host.getDiskAvailable() for host in self.env.hostlist]
        self.host_info.append(host_info)

    def save_workload_info(self, deployed, migrations):
        workload_info = dict()
        workload_info['interval'] = self.env.interval
        workload_info['total_containers'] = len(self.workload.created_containers)
        # check workload_info is empty or not
        if self.workload_info:
            workload_info['new_containers'] = workload_info['total_containers'] - self.workload_info[-1][
                'total_containers']
        else:
            workload_info['new_containers'] = workload_info['total_containers']
        workload_info['deployed'] = len(deployed)
        workload_info['migrations'] = len(migrations)
        workload_info['in_queue'] = len(self.workload.get_undeployed_containers())
        self.workload_info.append(workload_info)

    def save_container_info(self):
        container_info = dict()
        container_info['interval'] = self.env.interval
        container_info['active_containers'] = self.env.getNumActiveContainers()
        container_info['ips'] = [(c.getBaseIPS() if c else 0) for c in self.env.containerlist]
        container_info['apparent_ips'] = [(c.getApparentIPS() if c else 0) for c in self.env.containerlist]
        container_info['ram'] = [(c.getRAM() if c else 0) for c in self.env.containerlist]
        container_info['disk'] = [(c.getDisk() if c else 0) for c in self.env.containerlist]
        container_info['creation_ids'] = [(c.creationID if c else -1) for c in self.env.containerlist]
        container_info['host_alloc'] = [(c.getHostID() if c else -1) for c in self.env.containerlist]
        container_info['active'] = [(c.active if c else False) for c in self.env.containerlist]
        self.active_container_info.append(container_info)

    def save_all_container_info(self):
        container_info = dict()
        all_created_containers = [self.env.getContainerByCID(cid) for cid in
                                  list(np.where(self.workload.deployed_containers)[0])]
        container_info['interval'] = self.env.interval
        if self.datacenter.__class__.__name__ == 'Datacenter':
            container_info['application'] = [self.env.getContainerByCID(cid).application for cid in
                                             list(np.where(self.workload.deployed_containers)[0])]
        container_info['ips'] = [c.getBaseIPS() if c.active else 0 for c in all_created_containers]
        container_info['create'] = [c.createAt for c in all_created_containers]
        container_info['start'] = [c.startAt for c in all_created_containers]
        container_info['destroy'] = [c.destroyAt for c in all_created_containers]
        container_info['apparent_ips'] = [c.getApparentIPS() if c.active else 0 for c in all_created_containers]
        container_info['ram'] = [c.getRAM() if c.active else 0 for c in all_created_containers]
        container_info['disk'] = [c.getDisk() if c.active else 0 for c in all_created_containers]
        container_info['host_alloc'] = [c.getHostID() if c.active else -1 for c in all_created_containers]
        container_info['active'] = [c.active for c in all_created_containers]
        self.all_container_info.append(container_info)

    def save_metrics(self, destroyed, migrations):
        metrics = dict()
        metrics['interval'] = self.env.interval
        metrics['num_destroyed'] = len(destroyed)
        metrics['num_migrations'] = len(migrations)
        metrics['energy'] = [host.getPower() * self.env.intervaltime for host in self.env.hostlist]
        metrics['energy_total_interval'] = np.sum(metrics['energy'])
        metrics['energy_per_container_interval'] = np.sum(metrics['energy']) / self.env.getNumActiveContainers()
        metrics['response_time'] = [c.totalExecTime + c.totalMigrationTime for c in destroyed]
        metrics['avg_response_time'] = np.average(metrics['response_time']) if len(destroyed) > 0 else 0
        metrics['migration_time'] = [c.totalMigrationTime for c in destroyed]
        metrics['avg_migration_time'] = np.average(metrics['migration_time']) if len(destroyed) > 0 else 0
        metrics['sla_violations'] = len(np.where([c.destroyAt > c.sla for c in destroyed]))
        metrics['sla_violations_percentage'] = metrics['sla_violations'] * 100.0 / len(destroyed) if len(
            destroyed) > 0 else 0
        metrics['wait_time'] = [c.startAt - c.createAt for c in destroyed]
        metrics['energy_total_interval_pred'], metrics['avg_response_time_pred'] = self.run_simulation_GOBI()
        self.metrics.append(metrics)

    def save_scheduler_info(self, selected_containers, decision, scheduling_time):
        scheduler_info = dict()
        scheduler_info['interval'] = self.env.interval
        scheduler_info['selection'] = selected_containers
        scheduler_info['decision'] = decision
        scheduler_info['schedule'] = [(c.id, c.getHostID()) if c else (None, None) for c in self.env.containerlist]
        scheduler_info['scheduling_time'] = scheduling_time
        if self.datacenter.__class__.__name__ == 'Datacenter':
            scheduler_info['migration_time'] = self.env.intervalAllocTimings[-1]
        self.scheduler_info.append(scheduler_info)

    def save_fitness(self, fitness):
        energy_total_interval_pred = fitness[0][0].detach().item()
        avg_response_time_pred = fitness[0][1].detach().item()
        score = fitness[1].detach().item()
        metrics = self.metrics[-1]
        # metrics['energy_total_interval_score'] = energy_total_interval_pred * self._get_power_normalization()
        # metrics['avg_response_time_score'] = avg_response_time_pred * self._get_latency_normalization()
        metrics['energy_total_interval_score'] = energy_total_interval_pred * (
                self.energy_max - self.energy_min) + self.energy_min
        metrics['avg_response_time_score'] = avg_response_time_pred * (
                    self.latency_max - self.latency_min) + self.latency_min
        metrics['energy_total_interval_score_normal'] = energy_total_interval_pred
        metrics['avf_response_time_score_normal'] = avg_response_time_pred
        metrics['fitness'] = score

    def save_stats(self, deployed, migrations, destroyed, selected_containers, decision, scheduling_time, **kwargs):
        self.save_host_info()
        self.save_workload_info(deployed, migrations)
        self.save_container_info()
        self.save_all_container_info()
        self.save_metrics(destroyed, migrations)
        self.save_scheduler_info(selected_containers, decision, scheduling_time)
        if kwargs.get('fitness'):
            self.save_fitness(kwargs.get('fitness'))

    def _get_power_normalization(self):
        max_power = 0
        for h in self.env.hostlist:
            max_power += h.get_power_max()
        return max_power * self.env.intervaltime

    def _get_latency_normalization(self):
        metrics = self.metrics[-1]
        interval_response_time = metrics.get('response_time')
        if interval_response_time:
            self._max_response_time = max(self._max_response_time, max(interval_response_time))
            return self._max_response_time * len(interval_response_time)
        return 0

    def _temp_get_normalization(self):
        df = pd.read_csv('scheduler/BaGTI/datasets/energy_latency_' + str(len(self.env.hostlist)) + '_scheduling.csv')
        self.energy_max = df.iloc[:, -2].max()
        self.energy_min = df.iloc[:, -2].min()
        self.latency_max = df.iloc[:, -1].max()
        self.latency_min = 0

    def run_simple_simulation(self, decision):
        host_alloc = []
        container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)):
            host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1:
                host_alloc[c.getHostID()].append(c.id)
                container_alloc[c.id] = c.getHostID()
        decision = self.simulated_scheduler.filter_placement(decision)
        for cid, hid in decision:
            if self.env.getPlacementPossible(cid, hid) and container_alloc[cid] != -1:
                host_alloc[container_alloc[cid]].remove(cid)
                host_alloc[hid].append(cid)
        energy_total_interval_pred = 0
        for hid, cids in enumerate(host_alloc):
            ips = 0
            for cid in cids: ips += self.env.containerlist[cid].getApparentIPS()
            energy_total_interval_pred += self.env.hostlist[hid].getPowerFromIPS(ips)
        return energy_total_interval_pred * self.env.intervaltime, max(0, np.mean(
            [metric_d['avg_response_time'] for metric_d in self.metrics[-5:]]))

    def run_simulation_GOBI(self):
        host_alloc = []
        container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)):
            host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1:
                host_alloc[c.getHostID()].append(c.id)
                container_alloc[c.id] = c.getHostID()
        selected = self.simulated_scheduler.selection()
        decision = self.simulated_scheduler.filter_placement(self.simulated_scheduler.placement(selected))
        for cid, hid in decision:
            if self.env.getPlacementPossible(cid, hid) and container_alloc[cid] != -1:
                host_alloc[container_alloc[cid]].remove(cid)
                host_alloc[hid].append(cid)
        energy_total_interval_pred = 0
        for hid, cids in enumerate(host_alloc):
            ips = 0
            for cid in cids:
                # get ips并不是模拟host的ips，而是container目前所在host的ips，BUG
                ips += self.env.containerlist[cid].getApparentIPS()
            energy_total_interval_pred += self.env.hostlist[hid].getPowerFromIPS(ips)
        return energy_total_interval_pred * self.env.intervaltime, max(0, np.mean(
            [metric_d['avg_response_time'] for metric_d in self.metrics[-5:]]))

    ########################################################################################################

    def generate_graphs_with_interval(self, dirname, listinfo, obj, metric, metric2=None):
        fig, axes = plt.subplots(len(listinfo[0][metric]), 1, sharex=True, figsize=(4, 0.5 * len(listinfo[0][metric])))
        title = obj + '_' + metric + '_with_interval'
        total_intervals = len(listinfo)
        x = list(range(total_intervals))
        metric_with_interval = []
        metric2_with_interval = []
        ylimit = 0
        ylimit2 = 0
        for hostID in range(len(listinfo[0][metric])):  # todo change name on hostID
            metric_with_interval.append([listinfo[interval][metric][hostID] for interval in range(total_intervals)])
            ylimit = max(ylimit, max(metric_with_interval[-1]))
            if metric2:
                metric2_with_interval.append(
                    [listinfo[interval][metric2][hostID] for interval in range(total_intervals)])
                ylimit2 = max(ylimit2, max(metric2_with_interval[-1]))
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

    def generate_metrics_with_interval(self, dirname):
        fig, axes = plt.subplots(9, 1, sharex=True, figsize=(4, 5))
        x = list(range(len(self.metrics)))
        res = {}
        for i, metric in enumerate(['num_destroyed', 'num_migrations', 'energy_total_interval', 'avg_response_time',
                                    'avg_migration_time', 'sla_violations', 'sla_violations_percentage', 'wait_time',
                                    'energy_per_container_interval']):
            metric_with_interval = [self.metrics[i][metric] for i in
                                    range(len(self.metrics))] if metric != 'wait_time' else \
                [sum(self.metrics[i][metric]) for i in range(len(self.metrics))]
            axes[i].plot(x, metric_with_interval)
            axes[i].set_ylabel(metric, fontsize=5)
            axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
            res[metric] = sum(metric_with_interval)
            print("Summation ", metric, " = ", res[metric])
        print('Average energy (sum energy interval / sum num destroyed) = ',
              res['energy_total_interval'] / res['num_destroyed'])
        plt.tight_layout(pad=0)
        plt.savefig(dirname + '/' + 'Metrics' + '.pdf')
        plt.cla()
        plt.clf()
        plt.close('all')

    def generate_workload_with_interval(self, dirname):
        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(4, 5))
        x = list(range(len(self.workload_info)))
        for i, metric in enumerate(['total_containers', 'new_containers', 'deployed', 'migrations', 'in_queue']):
            metric_with_interval = [self.workload_info[i][metric] for i in range(len(self.workload_info))]
            axes[i].plot(x, metric_with_interval)
            axes[i].set_ylabel(metric)
            axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
        plt.tight_layout(pad=0)
        plt.savefig(dirname + '/' + 'Workload' + '.pdf')
        plt.cla()
        plt.clf()
        plt.close('all')

    ########################################################################################################

    def generate_complete_dataset(self, dirname, data, name):
        title = name + '_with_interval'
        metric_with_interval = []
        headers = list(data[0].keys())
        for datum in data:
            metric_with_interval.append([datum[value] for value in datum.keys()])
        df = pd.DataFrame(metric_with_interval, columns=headers)
        df.to_csv(dirname + '/' + title + '.csv', index=False)

    def generate_dataset_with_interval(self, dirname, metric, objfunc, metric2=None, objfunc2=None):
        title = metric + '_' + (metric2 + '_' if metric2 else "") + (objfunc + '_' if objfunc else "") + (
            objfunc2 + '_' if objfunc2 else "") + 'with_interval'
        total_intervals = len(self.host_info)
        metric_with_interval = []
        metric2_with_interval = []  # metric1 is of host and metric2 is of containers
        host_alloc_with_interval = []
        objfunc2_with_interval = []
        objfunc_with_interval = []
        for interval in range(total_intervals - 1):
            metric_with_interval.append(
                [self.host_info[interval][metric][hostID] for hostID in range(len(self.host_info[0][metric]))])
            host_alloc_with_interval.append([self.active_container_info[interval]['host_alloc'][cID] for cID in
                                             range(len(self.active_container_info[0]['host_alloc']))])  # todo rewrite
            objfunc_with_interval.append(self.metrics[interval + 1][objfunc])  # we get the destroy before simulation
            if metric2:
                metric2_with_interval.append(self.active_container_info[interval][metric2])
            if objfunc2:
                objfunc2_with_interval.append(self.metrics[interval + 1][objfunc2])
        df = pd.DataFrame(metric_with_interval)
        if metric2:
            df = pd.concat([df, pd.DataFrame(metric2_with_interval)], axis=1)
        df = pd.concat([df, pd.DataFrame(host_alloc_with_interval)], axis=1)
        df = pd.concat([df, pd.DataFrame(objfunc_with_interval)], axis=1)
        if objfunc2:
            df = pd.concat([df, pd.DataFrame(objfunc2_with_interval)], axis=1)
        df.to_csv(dirname + '/' + title + '.csv', header=False, index=False)

    # todo need to rewrite these two function
    def generate_dataset_with_interval2(self, dirname, metric, metric2, metric3, metric4, objfunc, objfunc2):
        title = metric + '_' + metric2 + '_' + metric3 + '_' + metric4 + '_' + objfunc + '_' + objfunc2 + '_' + 'with_interval'
        total_intervals = len(self.host_info)
        metric_with_interval = []
        metric2_with_interval = []
        metric3_with_interval = []
        metric4_with_interval = []
        host_alloc_with_interval = []
        objfunc2_with_interval = []
        objfunc_with_interval = []
        for interval in range(total_intervals - 1):
            metric_with_interval.append(
                [self.host_info[interval][metric][hostID] for hostID in range(len(self.host_info[0][metric]))])
            host_alloc_with_interval.append([self.active_container_info[interval]['host_alloc'][cID] for cID in
                                             range(len(self.active_container_info[0]['host_alloc']))])
            objfunc_with_interval.append(self.metrics[interval + 1][objfunc])
            metric2_with_interval.append(self.active_container_info[interval][metric2])
            metric3_with_interval.append(self.metrics[interval][metric3])
            metric4_with_interval.append(self.metrics[interval][metric4])
            objfunc2_with_interval.append(self.metrics[interval + 1][objfunc2])
        df = pd.DataFrame(metric_with_interval)
        df = pd.concat([df, pd.DataFrame(metric2_with_interval)], axis=1)
        df = pd.concat([df, pd.DataFrame(host_alloc_with_interval)], axis=1)
        df = pd.concat([df, pd.DataFrame(metric3_with_interval)], axis=1)
        df = pd.concat([df, pd.DataFrame(metric4_with_interval)], axis=1)
        df = pd.concat([df, pd.DataFrame(objfunc_with_interval)], axis=1)
        df = pd.concat([df, pd.DataFrame(objfunc2_with_interval)], axis=1)
        df.to_csv(dirname + '/' + title + '.csv', header=False, index=False)

    def generate_graphs(self, dirname):
        self.generate_graphs_with_interval(dirname, self.host_info, 'host', 'cpu')
        self.generate_graphs_with_interval(dirname, self.host_info, 'host', 'num_containers')
        self.generate_graphs_with_interval(dirname, self.host_info, 'host', 'power')
        self.generate_graphs_with_interval(dirname, self.host_info, 'host', 'base_ips', 'apparent_ips')  # test
        self.generate_graphs_with_interval(dirname, self.host_info, 'host', 'ips_cap', 'apparent_ips')
        self.generate_graphs_with_interval(dirname, self.active_container_info, 'container', 'ips', 'apparent_ips')
        self.generate_graphs_with_interval(dirname, self.active_container_info, 'container', 'host_alloc')
        self.generate_metrics_with_interval(dirname)
        self.generate_workload_with_interval(dirname)

    def generate_datasets(self, dirname):
        # self.generateDatasetWithInterval(dirname, 'cpu', objfunc='energytotalinterval')
        self.generate_dataset_with_interval(dirname, 'cpu', metric2='apparent_ips', objfunc='energy_total_interval',
                                            objfunc2='avg_response_time')
        self.generate_dataset_with_interval2(dirname, 'cpu', 'apparent_ips', 'energy_total_interval_pred',
                                             'avg_response_time_pred', objfunc='energy_total_interval',
                                             objfunc2='avg_response_time')

    def generate_complete_datasets(self, dirname):
        self.generate_complete_dataset(dirname, self.host_info, 'hostinfo')
        self.generate_complete_dataset(dirname, self.workload_info, 'workload_info')
        self.generate_complete_dataset(dirname, self.metrics, 'metrics')
        self.generate_complete_dataset(dirname, self.active_container_info, 'active_container_info')
        self.generate_complete_dataset(dirname, self.all_container_info, 'all_container_info')
        self.generate_complete_dataset(dirname, self.scheduler_info, 'scheduler_info')
