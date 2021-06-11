import numpy as np
from copy import deepcopy
from simulator.scenario import Scenario
from simulator.state import State
from policies.myopic import Myopic
from policies.adp import ADP
from utils import load_params

policies = {
    "myopic": Myopic,
    "adp": ADP
}


def run_simulation(scenario: Scenario, active_policy):
    simulation_history = []
    #ToDo: Generate state
    epoch = 0
    replica_state = State(scenario.init_blood_inventory, scenario.demands[epoch])
    while epoch < scenario.epochs - 1:  # Terminal test
        decisions = active_policy.get_action(replica_state)
        post_decision_state = replica_state.post_decision_state(decisions)
        if not post_decision_state:
            status = "INVALID_MOVE"
            break
        replica_state = replica_state.transition(post_decision_state,
                                                 donations=scenario.donations[epoch + 1],
                                                 demands=scenario.demands[epoch + 1])
        simulation_history.append(decisions)
        epoch += 1

    return replica_state.policy_reward, simulation_history, replica_state.id


def policy_evaluation(policy, scenarios):
    rewards = []
    gaps = []
    n = len(scenarios)
    for scenario in scenarios:
        reward, action_history, replica_id = run_simulation(scenario, policy)
        rewards.append(reward)
        if scenario.perfect_solution_reward:
            gaps.append(abs(reward - scenario.perfect_solution_reward)/scenario.perfect_solution_reward)
    return sum(rewards)/n, sum(gaps)/n


if __name__ == "__main__":
    params = {
        "policies": ["myopic"],
        "train_seed": 9874,
        "test_seed": 7383,
        "train_simulations": 100,
        "test_simulations": 100,
        "baseline_gap": False
    }
    params.update(load_params())

    # Simulation replicas (for fare comparison)
    train_generator = np.random.RandomState(seed=params['train_seed'])    # Move into policy.train()?
    test_generator = np.random.RandomState(seed=params['test_seed'])
    test_scenarios = [Scenario(index, test_generator, params) for index in range(params["test_simulations"])]
    for policy_name in params["policies"]:
        print(policy_name)
        policy = policies[policy_name](params)
        if policy.require_training:
            policy.train(train_generator)

        avg_reward, avg_gap = policy_evaluation(policy, test_scenarios)


