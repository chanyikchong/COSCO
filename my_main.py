from simulator.environment import AzureFog
from simulator.workload import BWGD2
import scheduler as sc  # time comsume
from my_package.my_simulator import MySimulator
from my_package.my_stats import MyStats


def initial_environment(num_host, container_limit, total_power, router_bw, arrival_rate, interval_time):
    datacenter = AzureFog(num_host)
    workload = BWGD2(arrival_rate)  # time consume
    scheduler = sc.GOBIScheduler('energy_latency_' + str(num_host))
    hostlist = datacenter.generate_hosts()
    env = MySimulator(total_power, router_bw, scheduler, container_limit, interval_time, hostlist)
    stats = MyStats()

    new_container_infos = workload.generate_new_containers(env.interval)
    deployed = env.add_containers_init(new_container_infos)
    decision = scheduler.placement(deployed)
    migrations = env.allocation(decision)
    env.execution()

    return datacenter, workload, scheduler, env, stats


def simulate_step(workload, scheduler, env, state, step):
    for i in range(step):
        new_container_infos = workload.generate_new_containers(env.interval)
        deployed = env.add_containers(new_container_infos)

        if i == step - 1:
            state.save_host_info(env.host_list)
            state.save_container_info(env.container_list)
            state.save_old_allocation(env.container_list)
        selected = scheduler.selection()
        decision = scheduler.filter_placement(scheduler.placement(selected + deployed))
        migrations = env.allocation(decision)
        if i == step - 1:
            state.save_new_allocation(env.container_list)
            env.finish_execution()
            workload.update_deployed_containers(env.get_creation_ids(migrations, deployed))
            destroyed = env.destroy_completed_containers()
            no_finish = [c for c in env.container_list if c]
            while len(no_finish) > 0:
                deployed = env.add_containers([])
                selected = scheduler.selection()
                decision = scheduler.filter_placement(scheduler.placement(selected + deployed))
                migrations = env.allocation(decision)
                env.finish_execution()
                workload.update_deployed_containers(env.get_creation_ids(migrations, deployed))
                destroyed.extend(env.destroy_completed_containers())
                no_finish = [c for c in env.container_list if c]
        else:
            env.execution()
            workload.update_deployed_containers(env.get_creation_ids(migrations, deployed))
            destroyed = env.destroy_completed_containers()
    print('in')


def main():
    num_hosts = 50
    container_limit = num_hosts
    total_power = 1000
    router_bw = 10000
    arrival_rate = 5
    interval_time = 300

    datacenter, workload, scheduler, env, stats = initial_environment(num_hosts, container_limit, total_power,
                                                                      router_bw, arrival_rate, interval_time)

    simulate_step(workload, scheduler, env, stats, 10)


if __name__ == '__main__':
    main()
