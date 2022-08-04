import sys
import logging as logger
import pickle
from os import rename
from time import time

# Simulator imports
from simulator.Simulator import *
from simulator.environment.AzureFog import *
from simulator.environment.BitbrainFog import *
from simulator.workload.BitbrainWorkload2 import *
from simulator.workload.Azure2017Workload import *
from simulator.workload.Azure2019Workload import *

# Scheduler imports
import scheduler as sc

# Auxiliary imports
from stats.Stats import Stats
from utils.Utils import *

# Global constants
NUM_SIM_STEPS = 200
HOSTS = 10 * 5
CONTAINERS = HOSTS
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300  # seconds
NEW_CONTAINERS = 0 if HOSTS == 10 else 3
DB_NAME = ''
DB_HOST = ''
DB_PORT = 0
HOSTS_IP = []
log_file = 'COSCO.log'
SAVE_FITNESS = False

if len(sys.argv) > 1:
    with open(log_file, 'w'):
        os.utime(log_file, None)


def initalize_environment():
    # Initialize simple fog datacenter
    ''' Can be SimpleFog, BitbrainFog, AzureFog // Datacenter '''
    datacenter = AzureFog(HOSTS)

    # Initialize workload
    ''' Can be SWSD, BWGD2, Azure2017Workload, Azure2019Workload // DFW, AIoTW '''
    workload = BWGD2(NEW_CONTAINERS, 1.5)

    # Initialize scheduler
    ''' Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI (arg = 'energy_latency_'+str(HOSTS)) '''
    scheduler = sc.GOBIScheduler('energy_latency_' + str(HOSTS))  # GOBIScheduler('energy_latency_'+str(HOSTS))
    # scheduler = RFScheduler()

    # Initialize Environment
    hostlist = datacenter.generateHosts()
    env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, CONTAINERS, INTERVAL_TIME, hostlist)

    # Initialize stats
    stats = Stats(env, workload, datacenter, scheduler)

    # Execute first step
    new_container_infos = workload.generateNewContainers(env.interval)  # New containers info

    deployed = env.addContainersInit(new_container_infos)  # Deploy new containers and get container IDs

    start = time()
    decision = scheduler.placement(deployed)  # Decide placement using container ids
    scheduling_time = time() - start

    if SAVE_FITNESS:
        migrations, fitness = env.simulationStep(decision, save_fitness=SAVE_FITNESS)  # Schedule containers
    else:
        migrations = env.simulationStep(decision, save_fitness=SAVE_FITNESS)
        fitness = None

    # Update workload allocated using creation IDs
    workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))

    print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
    print("Containers in host:", env.getContainersInHosts())
    print("Schedule:", env.getActiveContainerList())
    print_decision_and_migrations(decision, migrations)

    stats.save_stats(deployed, migrations, [], deployed, decision, scheduling_time, fitness=fitness)
    return datacenter, workload, scheduler, env, stats


def step_simulation(workload, scheduler, env, stats):
    new_container_infos = workload.generateNewContainers(env.interval)  # New containers info

    deployed, destroyed = env.addContainers(new_container_infos)  # Deploy new containers and get container IDs

    start = time()
    selected = scheduler.selection()  # Select container IDs for migration
    decision = scheduler.filter_placement(
        scheduler.placement(selected + deployed))  # Decide placement for selected container ids
    scheduling_time = time() - start

    if SAVE_FITNESS:
        migrations, fitness = env.simulationStep(decision, save_fitness=SAVE_FITNESS)  # Schedule containers
    else:
        migrations = env.simulationStep(decision, save_fitness=SAVE_FITNESS)
        fitness = None

    # Update workload deployed using creation IDs
    workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed))

    print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
    print("Deployed:", len(env.getCreationIDs(migrations, deployed)), "of", len(new_container_infos),
          [i[0] for i in new_container_infos])
    print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())
    print("Containers in host:", env.getContainersInHosts())
    print("Num active containers:", env.getNumActiveContainers())
    print("Host allocation:", [(c.getHostID() if c else -1) for c in env.containerlist])
    print_decision_and_migrations(decision, migrations)

    stats.save_stats(deployed, migrations, destroyed, selected, decision, scheduling_time, fitness=fitness)


def save_stats(stats, datacenter, workload, env, end=True):
    dirname = "logs/" + datacenter.__class__.__name__
    dirname += "_" + workload.__class__.__name__
    dirname += "_" + str(NUM_SIM_STEPS)
    dirname += "_" + str(HOSTS)
    dirname += "_" + str(CONTAINERS)
    dirname += "_" + str(TOTAL_POWER)
    dirname += "_" + str(ROUTER_BW)
    dirname += "_" + str(INTERVAL_TIME)
    dirname += "_" + str(NEW_CONTAINERS)
    if not os.path.exists("logs"):
        os.mkdir("logs")
    if os.path.exists(dirname):
        shutil.rmtree(dirname, ignore_errors=True)
    os.mkdir(dirname)
    stats.generate_datasets(dirname)
    if not end:
        return
    stats.generate_graphs(dirname)
    stats.generate_complete_datasets(dirname)
    stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
    with open(dirname + '/' + dirname.split('/')[1] + '.pk', 'wb') as handle:
        pickle.dump(stats, handle)


if __name__ == '__main__':
    import random
    import torch

    random.seed(10)
    torch.manual_seed(10)

    datacenter, workload, scheduler, env, stats = initalize_environment()

    for step in range(NUM_SIM_STEPS):
        print(color.BOLD + "Simulation Interval:", step, color.ENDC)
        step_simulation(workload, scheduler, env, stats)
        if env != '' and step % 10 == 0:
            save_stats(stats, datacenter, workload, env, end=False)

    save_stats(stats, datacenter, workload, env)
