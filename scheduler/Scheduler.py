import numpy as np
import pandas as pd

from utils.MathUtils import loess
from utils.MathConstants import LOCAL_REGRESSION_BANDWIDTH, LOCAL_REGRESSION_CPU_MULTIPLIER


class Scheduler:
    def __init__(self):
        self.env = None

    def set_environment(self, env):
        self.env = env

    def selection(self):
        pass

    def placement(self, container_list):
        pass

    def filter_placement(self, decision):
        filtered_decision = []
        for cid, hid in decision:
            if self.env.get_container_by_id(cid).get_host_id() != hid:
                filtered_decision.append((cid, hid))
        return filtered_decision

    def get_migration_from_host(self, host_id, decision):
        container_ids = []
        for (cid, _) in decision:
            hid = self.env.get_container_by_id(cid).get_host_id()
            if hid == host_id:
                container_ids.append(cid)
        return container_ids

    def get_migration_to_host(self, host_id, decision):
        container_ids = []
        for (cid, hid) in decision:
            if hid == host_id:
                container_ids.append(cid)
        return container_ids

    # Host selection
    def threshold_host_selection(self):
        selected_host_ids = []
        for i, host in enumerate(self.env.host_list):
            if host.get_cpu() > 70:
                selected_host_ids.append(i)
        return selected_host_ids

    def lr_selection(self, util_history):
        if len(util_history) < LOCAL_REGRESSION_BANDWIDTH:
            return self.threshold_host_selection()
        selected_host_ids = []
        x = list(range(LOCAL_REGRESSION_BANDWIDTH))
        for i, host in enumerate(self.env.host_list):
            host_l = [util_history[j][i] for j in range(len(util_history))]
            _, estimates = loess(x, host_l[-LOCAL_REGRESSION_BANDWIDTH:], poly_degree=1, alpha=0.6)
            weights = estimates['b'].values[-1]
            predicted_cpu = weights[0] + weights[1] * (LOCAL_REGRESSION_BANDWIDTH + 1)
            if LOCAL_REGRESSION_CPU_MULTIPLIER * predicted_cpu >= 100:
                selected_host_ids.append(i)
        return selected_host_ids

    def rlr_selection(self, util_history):
        if len(util_history) < LOCAL_REGRESSION_BANDWIDTH:
            return self.threshold_host_selection()
        selected_host_ids = list()
        x = list(range(LOCAL_REGRESSION_BANDWIDTH))
        for i, host in enumerate(self.env.host_list):
            host_l = [util_history[j][i] for j in range(len(util_history))]
            _, estimates = loess(x, host_l[-LOCAL_REGRESSION_BANDWIDTH:], poly_degree=1, alpha=0.6, robustify=True)
            weights = estimates['b'].values[-1]
            predicted_cpu = weights[0] + weights[1] * (LOCAL_REGRESSION_BANDWIDTH + 1)
            if LOCAL_REGRESSION_CPU_MULTIPLIER * predicted_cpu >= 100:
                selected_host_ids.append(i)
        return selected_host_ids

    def mad_selection(self, util_history):
        selected_host_ids = []
        for i, host in enumerate(self.env.host_list):
            host_l = [util_history[j][i] for j in range(len(util_history))]
            median_host_l = np.median(np.array(host_l))
            mad = np.median([abs(util_hst - median_host_l) for util_hst in host_l])
            threshold_cpu = 100 - LOCAL_REGRESSION_CPU_MULTIPLIER * mad
            utilized_cpu = host.get_cpu()
            if utilized_cpu > threshold_cpu:
                selected_host_ids.append(i)
        return selected_host_ids

    def iqr_selection(self, util_history):
        selected_host_ids = []
        for i, host in enumerate(self.env.host_list):
            host_l = [util_history[j][i] for j in range(len(util_history))]
            q1, q3 = np.percentile(np.array(host_l), [25, 75])
            iqr = q3 - q1
            threshold_cpu = 100 - LOCAL_REGRESSION_CPU_MULTIPLIER * iqr
            utilized_cpu = host.get_cpu()
            if utilized_cpu > threshold_cpu:
                selected_host_ids.append(i)
        return selected_host_ids

    # Container Selection

    def random_container_selection(self):
        selectable_ids = self.env.get_selectable_containers()
        if selectable_ids:
            return []
        selected_count = np.random.randint(0, len(selectable_ids)) + 1
        selected_ids = list()
        while len(selected_ids) < selected_count:
            id_choice = np.random.choice(selectable_ids)
            if self.env.container_list[id_choice]:
                selected_ids.append(id_choice)
                selectable_ids.remove(id_choice)
        return selected_ids

    def mmt_container_selection(self, selected_host_ids):
        selected_container_ids = []
        for hostID in selected_host_ids:
            container_ids = self.env.get_containers_of_host(hostID)
            ram_size = [self.env.container_list[cid].get_container_size() for cid in container_ids]
            if ram_size:
                mmt_container_id = container_ids[ram_size.index(min(ram_size))]
                selected_container_ids.append(mmt_container_id)
        return selected_container_ids

    def max_use_container_selection(self, selected_host_ids):
        selected_container_ids = []
        for hostID in selected_host_ids:
            container_ids = self.env.get_containers_of_host(hostID)
            if len(container_ids):
                container_ips = [self.env.container_list[cid].get_base_ips() for cid in container_ids]
                selected_container_ids.append(container_ids[container_ips.index(max(container_ips))])
        return selected_container_ids

    def max_cor_container_selection(self, selected_host_ids, util_history_container):
        selected_container_ids = []
        for host_id in selected_host_ids:
            container_ids = self.env.get_containers_of_host(host_id)
            if len(container_ids):
                host_l = [[util_history_container[j][cid] for j in range(len(util_history_container))] for cid in
                          container_ids]
                data = pd.DataFrame(host_l)
                data = data.T
                r_squared = []
                for i in range(data.shape[1]):
                    x = np.array(data.drop(data.columns[i], axis=1))
                    y = np.array(data.iloc[:, i])
                    x1 = np.c_[x, np.ones(x.shape[0])]
                    y_pred = np.dot(x1,
                                    np.dot(np.linalg.pinv(np.dot(np.transpose(x1), x1)), np.dot(np.transpose(x1), y)))
                    corr = np.corrcoef(np.column_stack((y, y_pred)), rowvar=False)
                    r_squared.append(corr[0][1] if not np.isnan(corr).any() else 0)
                selected_container_ids.append(container_ids[r_squared.index(max(r_squared))])
        return selected_container_ids

    # Container placement
    def random_placement(self, container_ids):
        decision = []
        for cid in container_ids:
            decision.append((cid, np.random.randint(0, len(self.env.host_list))))
        return decision

    def first_fit_placement(self, container_ids):
        decision = list()
        for cid in container_ids:
            for hostID in range(len(self.env.host_list)):
                if self.env.get_placement_possible(cid, hostID):
                    decision.append((cid, hostID))
                    break
        return decision

    def least_full_placement(self, container_ids):
        decision = []
        host_ipss = [(self.env.host_list[i].get_cpu(), i) for i in range(len(self.env.host_list))]
        for cid in container_ids:
            least_full_host = min(host_ipss)
            decision.append((cid, least_full_host[1]))
            if len(host_ipss) > 1:
                host_ipss.remove(least_full_host)
        return decision

    def max_full_placement(self, container_ids):
        decision = []
        host_ipss = [(self.env.host_list[i].get_cpu(), i) for i in range(len(self.env.host_list))]
        for cid in container_ids:
            max_full_host = max(host_ipss)
            decision.append((cid, max_full_host[1]))
            if len(host_ipss) > 1:
                host_ipss.remove(max_full_host)
        return decision
