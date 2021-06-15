import numpy as np
from copy import deepcopy
from simulator.scenario import Scenario
from simulator.state import State
from policies.policy import Policy
from policies.myopic import Myopic
from policies.adp import ADP
from policies.basic import Basic
from utils import load_params

policies = {
    "basic": Basic,
    "myopic": Myopic,
    "adp": ADP
}


def compute_reward(decisions: dict, blood_transfers: dict, transfer_rewards: dict):
    # decision (blood_type, age, (blood_type, surgery, substitution))
    value = 0
    committed_decisions = {k: v for k,v in decisions.items() if v > 0}
    for key, value in committed_decisions.items():
        supply_blood = key[0]
        demand_blood, surgery, allowed_substitution = key[2]
        if (not allowed_substitution and supply_blood != demand_blood) or \
                (allowed_substitution and not blood_transfers[(supply_blood, demand_blood)]):
            value += transfer_rewards["INFEASIBLE_SUBSTITUTION_PENALTY"]
        else:
            value += transfer_rewards["NO_SUBSTITUTION_BONUS"] if supply_blood == demand_blood \
                else transfer_rewards["SUBSTITUTION_PENALTY"]
            value += transfer_rewards["SUBSTITUTION_O-"] if supply_blood == "O-" else 0
            value += transfer_rewards["URGENT_DEMAND_BONUS"] if surgery == "urgent" \
                else transfer_rewards["ELECTIVE_DEMAND_BONUS"]
    return value


def run_simulation(scenario: Scenario, active_policy: Policy):
    simulation_history = []
    policy_reward = []
    #ToDo: Generate state
    epoch = 0
    replica_state = State(scenario.init_blood_inventory, scenario.demands[epoch])
    while epoch < scenario.epochs - 1:  # Terminal test
        decisions = active_policy.get_actions(replica_state)
        post_decision_state = replica_state.post_decision_state(decisions)
        if not post_decision_state:
            status = "INVALID_MOVE"
            break
        reward = compute_reward(decisions, scenario.blood_transfers, scenario.transfer_rewards)
        replica_state = replica_state.transition(post_decision_state,
                                                 next_donations=scenario.donations[epoch + 1],
                                                 next_demands=scenario.demands[epoch + 1])
        policy_reward.append(reward)
        simulation_history.append(decisions)
        epoch += 1

    print(f"Scenario: {scenario.index}")
    # print(simulation_history)
    return sum(policy_reward), simulation_history, scenario.index


def policy_evaluation(policy, scenarios):
    rewards = []
    gaps = []
    n = len(scenarios)
    for scenario in scenarios:
        reward, action_history, replica_id = run_simulation(scenario, policy)
        rewards.append(reward)
        if scenario.perfect_solution_reward:
            gaps.append(abs(reward - scenario.perfect_solution_reward)/scenario.perfect_solution_reward)
    return sum(rewards)/n, None  # sum(gaps)/n


if __name__ == "__main__":
    params = {
        "policies": ["basic"],
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
        print(f"Policy: {policy_name} | Avg. reward: {avg_reward} | gap: {avg_gap}")

