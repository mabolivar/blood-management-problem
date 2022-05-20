import copy

import numpy as np
import pandas as pd
import json
from datetime import datetime
from plotnine import *
from simulator.state import State


def simulate_solution(scenario, solution: list):
    epoch = 0
    replica_state = State(epoch, scenario.init_blood_inventory, scenario.demands[epoch])
    policy_reward = []
    post_decision_inventory = []
    post_decision_unsatisfied_demand = []
    while epoch < scenario.num_epochs:  # Terminal test
        decisions = solution[epoch]
        post_decision_state = replica_state.post_decision_state(decisions)
        if not post_decision_state:
            status = "INVALID_MOVE"
            print(f"epoch {epoch} - {status}")
            print(
                json.dumps(
                    {str(arc): str((replica_state.supply[arc[0]], value, post_decision_state[arc[0]]))
                     for arc, value in decisions.items()},
                    sort_keys=True, indent=4
                )
            )
            break
        reward = scenario.compute_reward(decisions)
        policy_reward.append(reward)
        post_decision_inventory.append(copy.deepcopy(post_decision_state))
        epoch += 1
        if epoch < scenario.num_epochs:
            replica_state = replica_state.transition(post_decision_state,
                                                     next_donations=scenario.donations[epoch],
                                                     next_demands=scenario.demands[epoch])

    return policy_reward, post_decision_inventory, post_decision_unsatisfied_demand


def plot_solution(scenario, scenario_index, network, decisions, policy_name='raw'):
    scenario_index = str(scenario_index).zfill(2)
    num_epochs = scenario.num_epochs
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
    demand_nodes = (pd.concat([pd.DataFrame.from_dict(demand, orient='index').assign(epoch=index) for index, demand in enumerate(scenario.demands)])
                    .reset_index()
                    .rename(columns={"index": "y_lab", 0: "b"})
                    .assign(type="demand")
                    [lambda x: x.b > 0]
                    )

    sink_node = nodes[lambda x: x.len == 0]

    nodes_to_plot = (pd.concat([supply_nodes, demand_nodes])
                     .assign(y_lab=lambda x: x.y_lab.astype(str))
                     )
    ### Arcs
    arcs = pd.DataFrame()
    arcs["tail_tuple"], arcs["head_tuple"] = zip(*network['arcs'])
    arcs = (arcs
            .assign(epoch=lambda x: x.tail_tuple.apply(lambda y: y[0]).astype(float),
                    next_epoch=lambda x: x.head_tuple.apply(lambda y: y[0]),
                    tail=lambda x: x.tail_tuple.apply(lambda y: y[1:]).astype(str),
                    head=lambda x: x.head_tuple.apply(lambda y: y[1:] if y != 'sink' else y).astype(str),
                    head_type=lambda x: np.where(x.head_tuple.apply(len) == 3, "supply", "demand"),
                    tail_type=lambda x: np.where(x.tail_tuple.apply(len) == 3, "supply", "demand")
                    )
            .assign(head_type=lambda x: np.where(x.next_epoch != "s",
                                                 x.head_type,
                                                 "sink"),
                    epoch=lambda x: x.epoch - 0.5 * (x.tail_type == 'supply'),
                    next_epoch=lambda x: np.where(x.next_epoch != "s",
                                                  x.next_epoch,
                                                  np.max(x.epoch) + 1).astype(float))
            .assign(next_epoch=lambda x: x.next_epoch - 0.5 * (x.head_type == 'supply'))
            )

    network_arcs = (arcs[lambda x: x.head_type != "sink"]
                    .filter(["epoch", "next_epoch", "tail", "head", "head_type"])
                    )

    # Solution arcs
    solution = (
        pd.concat([pd.DataFrame.from_dict(decision, orient='index')
                  .assign(epoch=index) for index, decision in enumerate(decisions)])
            .reset_index()
            .rename(columns={"index":"arc", 0: "flow"})
            .assign(tail=lambda x: x.arc.apply(lambda y: y[0]).astype(str),
                    head=lambda x: x.arc.apply(lambda y: y[1]).astype(str),
                    next_epoch=lambda x: x.epoch,
                    epoch=lambda x: x.epoch - 0.5,
                    line_type='supply->demand'
                    )
    )

    reward, inventory, unsatisfied_demand = simulate_solution(scenario, decisions)

    valid_inventory_nodes = nodes_to_plot[lambda x: x.type == 'supply'].y_lab.unique()
    inventory_arcs = (
        pd.concat(
            [pd.DataFrame.from_dict(units, orient='index').assign(epoch=i).reset_index()
             for i, units in enumerate(inventory)])
            .rename(columns={"index": "head_node", 0: "flow"})
            .assign(tail_node=lambda x: x.head_node.apply(lambda y: (y[0], y[1] - 1)),
                    epoch=lambda x: x.epoch - 0.5,
                    next_epoch=lambda x: x.epoch + 1,
                    tail=lambda x: x.tail_node.astype(str),
                    head=lambda x: x.head_node.astype(str),
                    line_type="supply->supply")
        [lambda x: (x.flow > 0) & x["head"].isin(valid_inventory_nodes)]
    )
    arcs_to_plot = (
        pd.concat([solution, inventory_arcs])
            .filter(["epoch", "next_epoch", "tail", "head", "flow", "line_type"])
            .assign(midx=lambda x: (x.epoch + x.next_epoch)/2)
    )

    label_size = 6
    p = (
            ggplot()
            + geom_point(data=nodes_to_plot,
                         mapping=aes(x="epoch", y="y_lab", color="type"))
            + geom_segment(data=network_arcs,
                           mapping=aes(x="epoch", y="tail",
                                       xend="next_epoch", yend="head",
                                       color="head_type"),
                           lineend="round",
                           arrow=arrow(length=0.02),
                           alpha=0.2)
            + geom_segment(data=arcs_to_plot,
                           mapping=aes(x="epoch", y="tail",
                                       xend="next_epoch", yend="head",
                                       linetype="line_type"),
                           lineend="round",
                           arrow=arrow(length=0.02))
            + geom_label(data=arcs_to_plot,
                         mapping=aes(x="midx", y="tail",
                                     label="flow"),
                         size=label_size)
            + geom_point(data=nodes_to_plot, mapping=aes(x="epoch", y="y_lab", color="type"))
            + geom_text(data=nodes_to_plot[lambda x: x.b != 0],
                        mapping=aes(x="epoch", y="y_lab", label="b"),
                        size=label_size)
            + scale_x_continuous(limits=[-0.5, num_epochs - 0.5])
            + labs(title=f"Scenario {scenario_index} - {policy_name}",
                   y="")
            + theme_xkcd()
    )

    p.save(filename=f"scenario_{scenario_index}_{policy_name}_graph_{now()}.jpg",
           path="figures/raw",
           format="jpg",
           width=16,
           height=10)

    return None


def now():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
