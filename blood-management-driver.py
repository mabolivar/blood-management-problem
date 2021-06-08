import numpy as np
from copy import deepcopy
from simulator.simulator import Simulator
from policies.myopic import Myopic
from policies.adp import ADP
from utils import load_params

policies = {
    "myopic": Myopic,
    "adp": ADP
}


def run_simulation(replica_state: Simulator, active_policy):
    simulation_history = []
    while not replica_state.terminal_test():
        action = active_policy.get_action(replica_state)
        replica_state = replica_state.result(action)
        simulation_history.append(action)

    return replica_state.policy_reward, simulation_history, replica_state.id


def policy_evaluation(policy, replicas):
    rewards = []
    gaps = []
    n = len(replicas)
    for replica in replicas:
        reward, action_history, replica_id = run_simulation(replica, policy)
        rewards.append(reward)
        if replica.perfect_solution_reward:
            gaps.append(abs(reward - replica.perfect_solution_reward)/replica.perfect_solution_reward)
    return sum(rewards)/n, sum(gaps)/n


if __name__ == "__main__":
    args = {
        "policies": ["myopic", "adp"],
        "train_seed": 9874,
        "test_seed": 7383,
        "train_simulations": 100,
        "test_simulations": 100,
        "baseline_gap": False
    }

    params = load_params()

    # Simulation replicas (for fare comparison)
    train_generator = np.random.RandomState(seed=args['train_seed'])    # Move into policy.train()?
    test_generator = np.random.RandomState(seed=args['test_seed'])
    test_replicas = [Simulator(index, test_generator, args) for index in range(args["test_simulations"])]
    for policy_name in args["policies"]:
        print(policy_name)
        policy = policies[policy_name](args)
        if policy.require_training:
            policy.train(train_generator)

        avg_reward, avg_gap = policy_evaluation(policy, deepcopy(test_replicas))


