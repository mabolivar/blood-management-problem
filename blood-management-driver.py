import os
import csv
import time
import numpy as np
from simulator.scenario import Scenario
from simulator.state import State
from policies.policy import Policy
from policies.myopic import Myopic
from policies.adp import VFA
from policies.basic import Basic
from parameters import load_params

policy_map = dict(basic=Basic, myopic=Myopic, adp=VFA)
OUTPUT_FOLDER = 'results'


def run_simulation(scenario: Scenario, active_policy: Policy):
    simulation_history = []
    policy_reward = []
    status = 'SOLVED'
    epoch = 0
    replica_state = State(epoch, scenario.init_blood_inventory, scenario.demands[epoch])
    start = time.time()
    while epoch < scenario.num_epochs:  # Terminal test
        decisions = active_policy.get_actions(replica_state, scenario.reward_map,
                                              scenario.allowed_blood_transfers)
        post_decision_state = replica_state.post_decision_state(decisions)
        if not post_decision_state:
            status = "INVALID_MOVE"
            break
        reward = scenario.compute_reward(decisions)
        policy_reward.append(reward)
        simulation_history.append(decisions)
        epoch += 1
        if epoch < scenario.num_epochs:
            replica_state = replica_state.transition(post_decision_state,
                                                     next_donations=scenario.donations[epoch],
                                                     next_demands=scenario.demands[epoch])
    end = time.time()
    if scenario.verbose:
        gap = round(100 * (scenario.perfect_solution_reward - sum(policy_reward))/scenario.perfect_solution_reward, 1)
        print(f"Policy: {active_policy.name}"
              f" - Scenario: {scenario.index} - Reward: {sum(policy_reward)} "
              f"- Perfect reward: {scenario.perfect_solution_reward} "
              f"- Status: {status}",
              f" - Gap: {gap}%")
    if scenario.to_visualize:
        scenario.export_solution(policy_name=active_policy.name,
                                 decisions=simulation_history)

    return sum(policy_reward), simulation_history, end - start


def policy_evaluation(policy, scenarios):
    performance = []    # Tuples with policy performance metrics
    rewards = []
    gaps = []
    n = len(scenarios)
    for scenario in scenarios:
        reward, action_history, execution_seconds = run_simulation(scenario, policy)
        rewards.append(reward)
        performance.append(
            (policy.name, scenario.index, reward, scenario.perfect_solution_reward, execution_seconds)
        )
        if scenario.perfect_solution_reward:
            gaps.append((scenario.perfect_solution_reward - reward)/scenario.perfect_solution_reward)
    return sum(rewards)/n, sum(gaps)/n, performance


if __name__ == "__main__":
    params = {
        "policies": ["myopic", "adp"],
        "test_seed": 7383,
        "test_simulations": 30,
        "verbose": True,
        "scenarios_to_visualize": 0,
        "instance_name": "epoch_15_age_3"
    }
    params.update(load_params(params["instance_name"]))

    # Simulation replicas (for fare comparison)
    test_generator = np.random.RandomState(seed=params['test_seed'])
    test_scenarios = [Scenario(index, test_generator, params) for index in range(params["test_simulations"])]
    policies_performance = [("policy", "scenario", "reward", "perfect_reward", "execution_secs")]
    for policy_name in params["policies"]:
        print(policy_name)
        policy = policy_map[policy_name](params)
        avg_reward, avg_gap, performance = policy_evaluation(policy, test_scenarios[::-1])
        policies_performance = policies_performance + performance
        print(f"Policy: {policy_name} | Avg. reward: {avg_reward} | gap: {(avg_gap * 100):.1f}%")

    # Export performance metrics
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_file = f"{OUTPUT_FOLDER}/performance_{params['test_simulations']}_{params['epochs']}_{params['max_age']}.csv"
    with open(output_file, "w") as out:
        csv_out = csv.writer(out, lineterminator='\n')
        for row in policies_performance:
            csv_out.writerow(row)
