import pandas as pd
from collections import defaultdict
from plotnine import *
from simulator.state import State


def get_demand_means (blood_types, default_value):
    demand_means = {k: default_value for k in blood_types}
    # Set here demand by blood type (for blood types that are different than the params['DEFAULT_VALUE_DIST'])
    demand_means['AB+'] = 3
    demand_means['B+'] = 9
    demand_means['O+'] = 18
    demand_means['B-'] = 2
    demand_means['AB-'] = 3
    demand_means['A-'] = 6
    demand_means['O-'] = 7
    demand_means['A+'] = 14

    demand_means['AB+'] = 0
    demand_means['B+'] = 0
    demand_means['O+'] = 0
    demand_means['B-'] = 0
    demand_means['AB-'] = 0
    demand_means['A-'] = 10
    demand_means['O-'] = 10
    demand_means['A+'] = 0

    demand_means['AB+'] = 3
    demand_means['B+'] = 9
    demand_means['O+'] = 18
    demand_means['B-'] = 2
    demand_means['AB-'] = 3
    demand_means['A-'] = 6
    demand_means['O-'] = 7
    demand_means['A+'] = 14

    return demand_means


def get_donation_means(blood_types, default_value):
    donation_means = {k: default_value for k in blood_types}

    # Set here donation by blood type (for blood types that are different than the params['DEFAULT_VALUE_DIST'])
    donation_means['AB+'] = 0
    donation_means['B+'] = 0
    donation_means['O+'] = 0
    donation_means['B-'] = 0
    donation_means['AB-'] = 0
    donation_means['A-'] = 10
    donation_means['O-'] = 10
    donation_means['A+'] = 0

    donation_means['AB+'] = 3
    donation_means['B+'] = 9
    donation_means['O+'] = 18
    donation_means['B-'] = 2
    donation_means['AB-'] = 3
    donation_means['A-'] = 6
    donation_means['O-'] = 7
    donation_means['A+'] = 14

    return donation_means

def load_params():
    params = defaultdict()
    params["epochs"] = 15
    params['blood_types'] = ['AB+', 'AB-', 'A+', 'A-', 'B+', 'B-', 'O+', 'O-']
    params["num_blood_types"] = len(params["blood_types"])
    params['allowed_transfers'] = [('AB+', 'AB+'), ('AB-', 'AB+'), ('AB-', 'AB-'), ('A+', 'AB+'), ('A+', 'A+'), ('A-', 'AB+'), ('A-', 'AB-'), ('A-', 'A+'), ('A-', 'A-'), ('B+', 'AB+'), ('B+', 'B+'), ('B-', 'AB+'), ('B-', 'AB-'), ('B-', 'B+'), ('B-', 'B-'), ('O+', 'AB+'), ('O+', 'A+'), ('O+', 'B+'), ('O+', 'O+'), ('O-', 'AB+'), ('O-', 'A+'), ('O-', 'B+'), ('O-', 'O+'), ('O-', 'AB-'), ('O-', 'A-'), ('O-', 'B-'), ('O-', 'O-')]

    params["blood_transfers"] = {(x, y): False for x in params['blood_types'] for y in params['blood_types']}
    for v in params["allowed_transfers"]:
        params["blood_transfers"][v] = True

    params["max_age"] = 3
    params['surgery_types'] = ['urgent', 'elective']
    params['substitution'] = [True]

    # Set here one step contribution function parameters  - BONUSES and PENALTIES
    params["transfer_rewards"] = {}
    params["transfer_rewards"]['INFEASIBLE_SUBSTITUTION_PENALTY'] = -50
    params["transfer_rewards"]['NO_SUBSTITUTION_BONUS'] = 0
    params["transfer_rewards"]['SUBSTITUTION_PENALTY'] = -10
    params["transfer_rewards"]['SUBSTITUTION_O-'] = -10
    params["transfer_rewards"]['URGENT_DEMAND_BONUS'] = 40
    params["transfer_rewards"]['ELECTIVE_DEMAND_BONUS'] = 20
    params["transfer_rewards"]['DISCARD_BLOOD_PENALTY'] = -10  # applied for the oldest age in the holding/vfa arcs

    # Set here max demand by blood type (when 'U'niform dist) or mean demand (when 'P'oisson dist)
    default_value = 20
    params['demand_means'] = get_demand_means(params["blood_types"], default_value)
    params['donation_means'] = get_donation_means(params["blood_types"], default_value)

    # The default weights to split the demand of a blood type is equal weights. The only requirement is that each
    # weight is positive and they add up to 1.
    # Default
    params['surgery_types_prop'] = {k: 1 / len(params['surgery_types']) for k in params['surgery_types']}
    params['substitution_prop'] = {k: 1 / len(params['substitution']) for k in params['substitution']}

    # Set here random surge parameters
    # params['time_periods_surge'] = set([4,8,10,12,14])
    params['time_periods_surge'] = set([i for i in range(1, params["epochs"]) if divmod(i, 3)[1] == 0])
    params['surge_prob'] = 0.7
    params['surge_factor'] = 6  # The surge demand is always going to be poisson with mean SURGE_FACTOR*demand_means, even if the regular demand distribution is Uniform
    params['surgery_types_prop']['urgent'] = 1 / 2
    params['surgery_types_prop']['elective'] = 1 - params['surgery_types_prop']['urgent']

    # Set here the weights for each substitution type (if different than the default)
    params['substitution_prop'][True] = 1
    # params['substitution_prop'][False] = 1 - params['substitution_prop'][True]

    # Move to simulation class
    #params['blood_inventory'] = None
    return params


def simulate_solution(scenario, solution: list):
    epoch = 0
    replica_state = State(scenario.init_blood_inventory, scenario.demands[epoch])
    policy_reward = []
    post_decision_inventory = [0]
    post_decision_unsatisfied_demand = []
    while epoch < scenario.num_epochs:  # Terminal test
        decisions = solution[epoch]
        post_decision_state = replica_state.post_decision_state(decisions)
        if not post_decision_state:
            status = "INVALID_MOVE"
            break
        reward = scenario.compute_reward(decisions)
        policy_reward.append(reward)
        post_decision_inventory.append(post_decision_state)
        epoch += 1

    return policy_reward, post_decision_inventory, post_decision_unsatisfied_demand


def plot_solution(scenario, network, decisions):
    nodes = (pd.DataFrame.from_dict(network['b'], orient="index")
             .reset_index()
             .rename(columns=({"index": "tuple_index", 0: "b"}), inplace=False)
             [lambda x: x.tuple_index != "sink"]
             .assign(name=lambda x: x.tuple_index.astype(str),
                     epoch=lambda x: pd.Series(x.tuple_index.apply(lambda y: y[0])).astype(float),
                     y_lab=lambda x: x.tuple_index.apply(lambda y: y[1:]),
                     blood_type=lambda x: x.tuple_index.apply(lambda y: y[1]),
                     len=lambda x: x.tuple_index.apply(len) * (x.name != "sink")
                     )
             .sort_values(by=["epoch", "blood_type"])
             )

    supply_nodes = (nodes
                    [lambda x: x.len == 3]
                    .assign(age=lambda x: list(zip(*x.tuple_index))[2],
                            epoch=lambda x: x.epoch - 0.5,
                            type="supply")
                    .filter(["epoch", "y_lab", "b", "type"])
                    )
    demand_nodes = (nodes
                    [lambda x: x.len > 3]
                    .assign(type="demand")
                    .filter(["epoch", "y_lab", "b", "type"])
                    )
    sink_node = nodes[lambda x: x.len == 0]

    nodes_to_plot = (pd.concat([supply_nodes, demand_nodes])
                     .assign(y_lab=lambda x: x.y_lab.astype(str))
                     )

    # Solution arcs
    solution = (
        pd.concat([pd.DataFrame.from_dict(decision, orient='index')
                  .assign(epoch=index) for index, decision in enumerate(decisions)])
            .reset_index()
            .rename(columns={"index":"arc", 0: "flow"})
            .assign(tail=lambda x: x.arc.apply(lambda y: y[0]).astype(str),
                    head=lambda x: x.arc.apply(lambda y: y[1]).astype(str),
                    next_epoch=lambda x: x.epoch + 1,
                    epoch=lambda x: x.epoch - 0.5,
                    line_type='supply->demand'
                    )
    )

    reward, inventory, unsatisfied_demand = simulate_solution(scenario, decisions)

    arcs_to_plot = (
        solution
        .filter(["epoch", "next_epoch", "tail", "head", "flow", "line_type"])
    )

    (
            ggplot()
            + geom_point(data=nodes_to_plot, mapping=aes(x="epoch", y="y_lab", color="type"))
            + geom_segment(data=arcs_to_plot,
                           mapping=aes(x="epoch", y="tail",
                                       xend="next_epoch", yend="head",
                                       linetype="line_type"),
                           lineend="round",
                           arrow=arrow(length=0.02))
            + geom_point(data=nodes_to_plot, mapping=aes(x="epoch", y="y_lab", color="type"))
    )

    return None

