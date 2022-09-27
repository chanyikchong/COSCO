import sys
import pickle
from time import time
import shutil
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

# Simulator imports
from simulator.Simulator import Simulator
import simulator.environment as environment
import simulator.workload as wl

# Scheduler imports
import scheduler as sc

# Auxiliary imports
from stats.Stats import Stats
from utils.Utils import *

# Global constants
log_file = 'COSCO.log'
SAVE_FITNESS = True

if len(sys.argv) > 1:
    with open(log_file, 'w'):
        os.utime(log_file, None)


def generate_dirname(**kwargs):
    sub_name = list()
    for k, v in kwargs.items():
        if not isinstance(v, str):
            try:
                v = str(v)
            except Exception as e:
                raise ValueError
        sub_name.append(v)
    dirname = "logs/%s" % "_".join(sub_name)
    return dirname


def initalize_environment(num_host, arrival_rate, total_power, router_bw, container_limit, interval_time):
    # Initialize simple fog datacenter
    # Can be SimpleFog, BitbrainFog, AzureFog // Datacenter
    datacenter = environment.AzureFog(num_host)

    # Initialize workload
    # Can be SWSD, BWGD2, Azure2017Workload, Azure2019Workload // DFW, AIoTW
    workload = wl.BWGD2(arrival_rate)

    # Initialize scheduler
    # Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI (arg = 'energy_latency_'+str(HOSTS))
    # scheduler = sc.GOBIScheduler('energy_latency_' + str(num_host))  # GOBIScheduler('energy_latency_'+str(HOSTS))
    scheduler = sc.GOBI2Scheduler('energy_latency_' + str(num_host))
    # scheduler = RFScheduler()

    # Initialize Environment
    hostlist = datacenter.generate_hosts()
    env = Simulator(total_power, router_bw, scheduler, container_limit, interval_time, hostlist)

    # Initialize stats
    stats = Stats(env, workload, datacenter, scheduler)

    # Execute first step
    new_container_infos = workload.generate_new_containers(env.interval)  # New containers info

    deployed = env.add_containers_init(new_container_infos)  # Deploy new containers and get container IDs

    start = time()
    decision = scheduler.placement(deployed)  # Decide placement using container ids
    scheduling_time = time() - start

    # check decision and get possible migration
    migrations = env.allocation(decision)
    # save the input data for training
    stats.save_input_stats(deployed, migrations)
    stats.save_scheduler_info(deployed, decision, scheduling_time)

    # scheduler fitness for the final placement
    fitness = scheduler.allocation_fitness()
    # execute tasks
    env.execution()

    # Update workload allocated using creation IDs
    workload.update_deployed_containers(env.get_creation_ids(migrations, deployed))

    print("Deployed containers' creation IDs:", env.get_creation_ids(migrations, deployed))
    print("Deployed:", len(env.get_creation_ids(migrations, deployed)), "of", len(new_container_infos),
          [i[0] for i in new_container_infos])

    stats.save_energy()
    # destroyed finished tasks
    destroyed = env.destroy_completed_containers()
    # save statistical metrics
    stats.save_metrics(destroyed, migrations)
    stats.save_fitness(fitness)

    print("Destroyed:", len(destroyed), "of", env.get_num_active_containers())
    print("Containers in host:", env.get_containers_in_hosts())
    print("Num active containers:", env.get_num_active_containers())
    print("Host allocation:", [(c.get_host_id() if c else -1) for c in env.container_list])
    print_decision_and_migrations(decision, migrations)

    return datacenter, workload, scheduler, env, stats


def step_simulation(workload, scheduler, env, stats):
    new_container_infos = workload.generate_new_containers(env.interval)  # New containers info

    deployed = env.add_containers(new_container_infos)

    start = time()
    selected = scheduler.selection()  # Select container IDs for migration
    decision = scheduler.filter_placement(
        scheduler.placement(selected + deployed))  # Decide placement for selected container ids
    scheduling_time = time() - start

    migrations = env.allocation(decision)
    stats.save_input_stats(deployed, migrations)
    stats.save_scheduler_info(deployed, decision, scheduling_time)

    fitness = scheduler.allocation_fitness()
    env.execution()

    # Update workload deployed using creation IDs
    workload.update_deployed_containers(env.get_creation_ids(migrations, deployed))
    print("Deployed containers' creation IDs:", env.get_creation_ids(migrations, deployed))
    print("Deployed:", len(env.get_creation_ids(migrations, deployed)), "of", len(new_container_infos),
          [i[0] for i in new_container_infos])

    stats.save_energy()
    destroyed = env.destroy_completed_containers()
    stats.save_metrics(destroyed, migrations)
    stats.save_fitness(fitness)

    print("Destroyed:", len(destroyed), "of", env.get_num_active_containers())
    print("Containers in host:", env.get_containers_in_hosts())
    print("Num active containers:", env.get_num_active_containers())
    print("Host allocation:", [(c.get_host_id() if c else -1) for c in env.container_list])

    print_decision_and_migrations(decision, migrations)


def save_stats(dirname, stats, end=True):
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


def static_main():
    # random.seed(10)
    # torch.manual_seed(10)

    sim_steps = 1000
    num_hosts = 10 * 5
    container_limit = num_hosts
    total_power = 1000
    router_bw = 10000
    interval_time = 300  # seconds
    arrival_rate = 0 if num_hosts == 10 else 5

    datacenter, workload, scheduler, env, stats = initalize_environment(num_hosts, arrival_rate, total_power, router_bw,
                                                                        container_limit, interval_time)

    dirname = generate_dirname(datacenter=datacenter.__class__.__name__, workload=workload.__class__.__name__,
                               num_steps=sim_steps, num_host=num_hosts, container_limit=container_limit,
                               total_power=total_power, router_bw=router_bw, interval_time=interval_time,
                               arrival_rate=arrival_rate, dynamic='static')

    for step in range(sim_steps):
        print(Color.BOLD + "Simulation Interval:", step, Color.ENDC)
        step_simulation(workload, scheduler, env, stats)
        if env != '' and step % 10 == 0:
            save_stats(dirname, stats, end=False)

    save_stats(dirname, stats)


def dynamic_main():
    # random.seed(10)
    # torch.manual_seed(10)

    sim_steps = 1500
    num_hosts = 10 * 5
    container_limit = num_hosts
    total_power = 1000
    router_bw = 10000
    interval_time = 300  # seconds
    arrival_rate = 0 if num_hosts == 10 else 3
    change_probability = 0.002
    change_step = 600

    arrival_rate_list = [1, 2, 3, 4, 5, 6, 7]
    arrival_rate_list.remove(arrival_rate)
    num_job_class = [3, 5, 7, 9]

    datacenter, workload, scheduler, env, stats = initalize_environment(num_hosts, arrival_rate, total_power, router_bw,
                                                                        container_limit, interval_time)

    dirname = generate_dirname(datacenter=datacenter.__class__.__name__, workload=workload.__class__.__name__,
                               num_steps=sim_steps, num_host=num_hosts, container_limit=container_limit,
                               total_power=total_power, router_bw=router_bw, interval_time=interval_time,
                               arrival_rate=arrival_rate, dynamic="dynamic")

    keep_ar = list()

    for step in range(sim_steps):
        if step > change_step:
            change_step = np.inf
        # if random.random() < change_probability:
            new_ar = random.choice(arrival_rate_list)
            arrival_rate_list.remove(new_ar)
            arrival_rate_list.append(arrival_rate)
            arrival_rate = new_ar
            workload.set_arrival_rate(arrival_rate)
        print(Color.BOLD + "Simulation Interval:", step, Color.ENDC)
        step_simulation(workload, scheduler, env, stats)
        keep_ar.append(arrival_rate)
        if env != '' and step % 10 == 0:
            save_stats(dirname, stats, end=False)

    save_stats(dirname, stats)
    plt.plot(range(len(keep_ar)), keep_ar)
    plt.show()


if __name__ == '__main__':
    static_main()
    # dynamic_main()
