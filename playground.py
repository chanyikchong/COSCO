import os
import argparse
import pickle
import random
import logging
import shutil
from time import time
import matplotlib.pyplot as plt

import torch
import pandas as pd
from memory_profiler import profile

from simulator.environment.AzureFog import AzureFog
from simulator.workload.BitbrainWorkload2 import BWGD2
from simulator.Simulator import Simulator
from stats.Stats import Stats
from utils.Utils import generate_decision_migration_string
from utils.ColorUtils import color

# from scheduler.IQR_MMT_Random import IQRMMTRScheduler
from scheduler.MAD_MMT_Random import MADMMTRScheduler
from scheduler.MAD_MC_Random import MADMCRScheduler
from scheduler.LR_MMT_Random import LRMMTRScheduler
# from scheduler.Random_Random_FirstFit import RFScheduler
# from scheduler.Random_Random_LeastFull import RLScheduler
# from scheduler.RLR_MMT_Random import RLRMMTRScheduler
# from scheduler.Threshold_MC_Random import TMCRScheduler
# from scheduler.Random_Random_Random import RandomScheduler
# from scheduler.HGP_LBFGS import HGPScheduler
# from scheduler.GA import GAScheduler
from scheduler.GOBI import GOBIScheduler


# from scheduler.GOBI2 import GOBI2Scheduler
# from scheduler.DRL import DRLScheduler
# from scheduler.DQL import DQLScheduler
# from scheduler.POND import PONDScheduler
# from scheduler.SOGOBI import SOGOBIScheduler
# from scheduler.SOGOBI2 import SOGOBI2Scheduler
# from scheduler.HGOBI import HGOBIScheduler
# from scheduler.HGOBI2 import HGOBI2Scheduler
# from scheduler.HSOGOBI import HSOGOBIScheduler
# from scheduler.HSOGOBI2 import HSOGOBI2Scheduler


def init_logger(file_name, level='debug'):
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()

    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(eval("logging.%s" % level.upper()))
    return logger


def initalize_environment(env_setting, scheduler, logger):
    datacenter = AzureFog(env_setting.get('num_hosts'))
    workload = BWGD2(env_setting.get('mean_container'), env_setting.get('std_container'))

    host_list = datacenter.generateHosts()

    env = Simulator(env_setting.get('total_power'), env_setting.get('router_bw'), scheduler,
                    env_setting.get('max_container'), env_setting.get('interval_time'), host_list)

    # Initialize stats
    stats = Stats(env, workload, datacenter, scheduler)

    # Execute first step
    new_container_infos = workload.generateNewContainers(env.interval)  # New containers info
    deployed = env.addContainersInit(new_container_infos)  # Deploy new containers and get container IDs
    start = time()
    decision = scheduler.placement(deployed)  # Decide placement using container ids
    scheduling_time = time() - start
    migrations = env.allocateInit(decision)  # Schedule containers
    workload.updateDeployedContainers(
        env.getCreationIDs(migrations, deployed))  # Update workload allocated using creation IDs
    logger.info("Deployed containers' creation IDs: %s" % str(env.getCreationIDs(migrations, deployed)))
    logger.info("Containers in host: %s" % str(env.getContainersInHosts()))
    logger.info("Schedule: %s" % str(env.getActiveContainerList()))
    logger.info(generate_decision_migration_string(decision, migrations))

    stats.saveStats(deployed, migrations, [], deployed, decision, scheduling_time)
    return datacenter, workload, scheduler, env, stats


def step_simulation(workload, scheduler, env, stats, logger):
    new_container_infos = workload.generateNewContainers(env.interval)  # New containers info
    deployed, destroyed = env.addContainers(new_container_infos)  # Deploy new containers and get container IDs
    start = time()
    selected = scheduler.selection()  # Select container IDs for migration

    # Decide placement for selected container ids
    decision = scheduler.filter_placement(scheduler.placement(selected + deployed))
    scheduling_time = time() - start
    migrations = env.simulationStep(decision)  # Schedule containers

    # Update workload deployed using creation IDs
    workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))
    logger.info("Deployed containers' creation IDs: %s", str(env.getCreationIDs(migrations, deployed)))
    logger.info("Deployed: %d of %d %s" % (len(env.getCreationIDs(migrations, deployed)), len(new_container_infos),
                                           str([i[0] for i in new_container_infos])))
    logger.info("Destroyed: %d of %d" % (len(destroyed), env.getNumActiveContainers()))
    logger.info("Containers in host: %s" % str(env.getContainersInHosts()))
    logger.info("Num active containers: %s", str(env.getNumActiveContainers()))
    logger.info("Host allocation: %s" % str([(c.getHostID() if c else -1) for c in env.containerlist]))
    logger.info(generate_decision_migration_string(decision, migrations))

    stats.saveStats(deployed, migrations, destroyed, selected, decision, scheduling_time)


def save_stats(log_file, env_setting, stats, datacenter, workload, env, end=True):
    dirname = "logs/" + datacenter.__class__.__name__
    dirname += "_" + workload.__class__.__name__
    for v in env_setting.values():
        dirname += "_" + str(v)
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if os.path.exists(dirname):
        shutil.rmtree(dirname, ignore_errors=True)
    os.mkdir(dirname)
    stats.generateDatasets(dirname)
    if 'Datacenter' in datacenter.__class__.__name__:
        saved_env = stats.env
        saved_workload = stats.workload
        saved_datacenter = stats.datacenter
        saved_scheduler = stats.scheduler
        saved_sim_scheduler = stats.simulated_scheduler
        stats.env = None
        stats.workload = None
        stats.datacenter = None
        stats.scheduler = None
        stats.simulated_scheduler = None
        with open(dirname + '/' + dirname.split('/')[1] + '.pk', 'wb') as handle:
            pickle.dump(stats, handle)
        stats.env = saved_env
        stats.workload = saved_workload
        stats.datacenter = saved_datacenter
        stats.scheduler = saved_scheduler
        stats.simulated_scheduler = saved_sim_scheduler
    if not end:
        return
    stats.generateGraphs(dirname)
    stats.generateCompleteDatasets(dirname)
    stats.env = None
    stats.workload = None
    stats.datacenter = None
    stats.scheduler = None
    if 'Datacenter' in datacenter.__class__.__name__:
        stats.simulated_scheduler = None
        logger.handlers.clear()
        env.logger.handlers.clear()
        if os.path.exists(dirname + '/' + log_file):
            os.remove(dirname + '/' + log_file)
        os.rename(log_file, dirname + '/' + log_file)
    with open(dirname + '/' + dirname.split('/')[1] + '.pk', 'wb') as handle:
        pickle.dump(stats, handle)
    return dirname

def test(log_file):
    with open('var/env_setting.pkl', 'rb') as f:
        env_setting = pickle.load(f)
    with open('var/stats.pkl', 'rb') as f:
        stats = pickle.load(f)
    with open('var/datacenter.pkl', 'rb') as f:
        datacenter = pickle.load(f)
    with open('var/workload.pkl', 'rb') as f:
        workload = pickle.load(f)
    with open('var/env.pkl', 'rb') as f:
        env = pickle.load(f)
    
    start_time = time()
    dirname = save_stats(log_file, env_setting, stats, datacenter, workload, env)
    if os.path.exists(file_pandas):
        f_pd = pd.read_csv(file_pandas)
    else:
        f_pd = pd.DataFrame(columns=list(env_setting.keys()))
    row = pd.Series(env_setting)
    row['dirname'] = dirname
    f_pd = f_pd.append(row, ignore_index=True)
    f_pd = f_pd.drop_duplicates(keep='last')
    f_pd.to_csv(file_pandas, index=False)


def main(env_setting, scheduler_dict, mean_container_list, num_seed=10, start_seed=10):
    logger = init_logger(log_file, log_level)
    for mean_container in mean_container_list:
        env_setting['mean_container'] = mean_container
        for seed in range(start_seed, start_seed + num_seed):

            env_setting['seed'] = seed
            random.seed(env_setting.get('seed'))
            torch.manual_seed(env_setting.get('seed'))
            for scheduler_key, scheduler_value in scheduler_dict.items():

                env_setting['scheduler'] = scheduler_key
                scheduler = eval("%s('%s')" % (scheduler_key, scheduler_value)) if len(scheduler_value) > 0 else eval(
                    "%s()" % scheduler_key)

                datacenter, workload, scheduler, env, stats = initalize_environment(env_setting, scheduler, logger)

                for step in range(env_setting.get('num_step')):
                    logger.info("%sSimulation Interval: %s%s" % (color.BOLD, step, color.ENDC))
                    step_simulation(workload, scheduler, env, stats, logger)

                dirname = save_stats(log_file, env_setting, stats, datacenter, workload, env)

                if os.path.exists(file_pandas):
                    f_pd = pd.read_csv(file_pandas)
                else:
                    f_pd = pd.DataFrame(columns=list(env_setting.keys()))
                row = pd.Series(env_setting)
                row['dirname'] = dirname
                f_pd = f_pd.append(row, ignore_index=True)
                f_pd = f_pd.drop_duplicates(keep='last')
                f_pd.to_csv(file_pandas, index=False)
                del scheduler
                del datacenter
                del workload
                del env
                del stats
                del f_pd
                del row


if __name__ == '__main__':
    plt.style.use(['science'])
    plt.rcParams["text.usetex"] = False
    file_pandas = 'logs/file_df.csv'

    log_file = 'COSCO.log'
    log_level = 'info'

    env_setting = {'num_hosts': 50,
                   'max_container': 50,
                   'mean_container': 5,
                   'std_container': 1.5,
                   'total_power': 1000,
                   'router_bw': 10000,
                   'interval_time': 300,
                   'num_step': 100,
                   'seed': 10}

    scheduler_dict = {'GOBIScheduler': 'energy_latency_%s' % env_setting.get("num_hosts"),
                      'LRMMTRScheduler': '',
                      'MADMMTRScheduler': ''}
    # scheduler_dict = {'GOBIScheduler': 'energy_latency_%s' % env_setting.get("num_hosts")}
    # scheduler_dict = {'MADMMTRScheduler': ''}
    # scheduler_dict = {'MADMMTRScheduler': ''}
    mean_container_list = [6, 7, 8, 9]

    main(env_setting, scheduler_dict, mean_container_list, 10, 10)
    # test(log_file)
