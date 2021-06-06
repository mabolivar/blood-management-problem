from copy import deepcopy
from simulator.simulator import Simulator
from policies.myopic import Myopic
from policies.adp import ADP


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
    params = {
        "policies": ["myopic", "adp"],
        "seed": 234,
        "epochs": 50,
        "train_simulations": 100,
        "test_simulations": 100,
        "baseline_gap": False
    }

    # Simulation replicas (for fare comparison)
    replicas = [Simulator(index, params) for index in range(params["test_simulations"])]
    for policy_name in params["policies"]:
        print(policy_name)
        policy = policies[policy_name](params)
        if policy.require_training:
            policy.train()

        avg_reward, avg_gap = policy_evaluation(policy, deepcopy(replicas))


