from numpy.random import RandomState
from copy import deepcopy
from ortools.linear_solver import pywraplp


class Scenario(object):
    def __init__(self, index, rnd_generator: RandomState, params):
        self.index = index
        self.rnd_generator = rnd_generator
        self.num_epochs = params['epochs']
        self.epochs = list(range(self.num_epochs))

        self.blood_types = params["blood_types"]
        self.max_blood_age = params['max_age']
        self.blood_ages = list(range(self.max_blood_age))
        self.donation_means = params["donation_means"]
        self.demand_means = params["donation_means"]

        self.allowed_blood_transfers = params["blood_transfers"]
        self.transfer_rewards = params["transfer_rewards"]
        self.time_periods_surge = params["time_periods_surge"]
        self.surge_prob = params["surge_prob"]
        self.surge_factor = params['surge_factor']
        self.surgery_types = params['surgery_types']
        self.surgery_types_prop = params['surgery_types_prop']
        self.substitution = params["substitution"]
        self.substitution_prop = params['substitution_prop']
        self.demand_types = [(i, j, k) for i in self.blood_types for j in self.surgery_types for k in self.substitution]

        self.demands = self.generate_demands(self.num_epochs)
        self.donations = self.generate_donations(self.num_epochs)

        self.blood_groups = [(i, j) for i in self.blood_types for j in self.blood_ages]
        self.init_blood_inventory = self.generate_init_blood_inventory()

        self.reward_map = self.get_epoch_reward_map()
        self.perfect_solution_reward, self.perfect_solution = self.get_perfect_information_solution()

    def generate_demands(self, epochs):
        demand = []
        for t in range(epochs):
            factor = self.surge_factor \
                if t in self.time_periods_surge and self.rnd_generator.uniform(0, 1) < self.surge_prob else 1

            demand.append({(blood, surgery, substitution): int(self.rnd_generator.poisson(
                    factor * self.demand_means[blood] * self.surgery_types_prop[surgery] *
                    self.substitution_prop[substitution])) for blood, surgery, substitution in self.demand_types})

        return demand

    def generate_donations(self, epochs):
        return [{i: 0 for i in self.blood_types}] + \
               [{i: int(self.rnd_generator.poisson(self.donation_means[i])) for i in self.blood_types}
                for _ in range(1, epochs)]

    def generate_init_blood_inventory(self):
        multiplier = {age: .9 if age == 0 else (0.1 / (self.max_blood_age - 1)) for age in range(self.max_blood_age)}
        return {(blood_type, age): int(self.rnd_generator.poisson(self.donation_means[blood_type]) * multiplier[age])
                for blood_type, age in self.blood_groups}

    def get_epoch_reward_map(self):
        # map keys = ((blood_type, age), (blood_type, surgery, substitution))
        reward_map = {(s, d): 0 for s in self.blood_groups for d in self.demand_types
                      if self.allowed_blood_transfers[(s[0], d[0])] and (d[2] or s[0] == d[0])}
        for s, d in reward_map.keys():
            supply_blood = s[0]
            demand_blood = d[0]
            surgery = d[1]
            reward_map[(s, d)] += self.transfer_rewards["NO_SUBSTITUTION_BONUS"] if supply_blood == demand_blood \
                else self.transfer_rewards["SUBSTITUTION_PENALTY"]
            reward_map[(s, d)] += self.transfer_rewards["SUBSTITUTION_O-"] if supply_blood == "O-" else 0
            reward_map[(s, d)] += self.transfer_rewards["URGENT_DEMAND_BONUS"] if surgery == "urgent" \
                else self.transfer_rewards["ELECTIVE_DEMAND_BONUS"]

        return reward_map

    def compute_reward(self, decisions):
        return sum(self.reward_map.get(decision, self.transfer_rewards["INFEASIBLE_SUBSTITUTION_PENALTY"])
                   for decision, units in decisions.items() if units > 0)

    def get_perfect_information_solution(self):

        # Build network
        last_epoch = self.num_epochs - 1
        max_blood_age = max(self.blood_ages)

        supply_demand_arcs_per_epoch = [((t,) + s, (t,) + d) for t in self.epochs for s in self.blood_groups
                                        for d in self.demand_types if self.allowed_blood_transfers[s[0], d[0]] and (d[2] or s[0] == d[0])] # ToDo: Replacement is missing
        supply_supply_arcs_per_epoch = [((t,) + s, (t+1, s[0], s[1] + 1)) for t in self.epochs[:-1]
                                        for s in self.blood_groups
                                        if s[1] + 1 <= max_blood_age]
        demand_sink_arcs_per_epoch = [((t,) + d, "sink") for t in self.epochs for d in self.demand_types]
        supply_sink_arcs_per_epoch = [((t, blood_type, max_blood_age), "sink")
                                      for t in self.epochs
                                      for blood_type in self.blood_types]
        supply_sink_arcs_last_epoch = [((last_epoch,) + s, "sink") for s in self.blood_groups if s[1] != max_blood_age]

        arcs = (supply_demand_arcs_per_epoch +
                supply_supply_arcs_per_epoch +
                demand_sink_arcs_per_epoch +
                supply_sink_arcs_per_epoch +
                supply_sink_arcs_last_epoch)
        arcs.sort(key=lambda x: (x[0][0], x[0][1], str(x[0][2])))

        head_nodes, tail_nodes = [set(nodes) for nodes in zip(*arcs)]
        nodes = set.union(head_nodes, tail_nodes)
        nodes_per_epoch = self.blood_groups + self.demand_types

        # Adjacency dictionary for regular epoch
        inner_arcs = {node: set() for node in nodes}
        outer_arcs = {node: set() for node in nodes}
        for arc in arcs:
            outer_arcs[arc[0]].add(arc)
            inner_arcs[arc[1]].add(arc)

        # Supply/demand for nodes
        sink_demand = -1 * (sum(self.init_blood_inventory.values())
                          + sum(sum(donation.values()) if donation is not None else 0
                                for donation in self.donations))
        b = {node: 0 for node in nodes}
        b['sink'] = sink_demand
        
        # Initial supply for blood groups
        for blood_group, inventory in self.init_blood_inventory.items():
            b[(0,) + blood_group] = inventory
        # Add donations to supply nodes
        for t in self.epochs[1:]:
            for blood_type in self.blood_types:
                b[(t, blood_type, 0)] = self.donations[t][blood_type]

        upper_bound = {arc: self.demands[arc[0][0]][(arc[0][1:])] for arc in demand_sink_arcs_per_epoch}
        rewards = {arc: self.reward_map[(arc[0][1:], arc[1][1:])] for arc in supply_demand_arcs_per_epoch}

        # Build model
        solver = pywraplp.Solver('simple_mip_program',
                                 pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # A dictionary vars created to contain the referenced variables
        x = {(head, tail): solver.NumVar(lb=0, ub=upper_bound.get((head, tail), solver.Infinity()),
                                         name=f'x_{head}_{tail}') for head, tail in arcs}

        # Balance constraint
        for i in nodes:
            solver.Add(
                sum(x[outer_arc] for outer_arc in outer_arcs[i]) -
                sum(x[inner_arc] for inner_arc in inner_arcs[i]) == b[i],
                name='c_' + str(i)
            )

        # Objective function
        reward_function = [x[arc] * reward for arc, reward in rewards.items()]
        solver.Maximize(solver.Sum(reward_function))

        if False:
            print(solver.ExportModelAsLpFormat(True))

        solver_status = solver.Solve()
        # ToDO: Check '-AB' supply on epoch 0
        if False and solver_status == pywraplp.Solver.OPTIMAL:
            for t in self.epochs:
                for blood_type in self.blood_types:
                    for age in self.blood_ages:
                        for demand_node in self.demand_types:
                            v = x.get(((t, blood_type, age), (t,) + demand_node), None)
                            if v and v.solution_value() >= 0.999:
                                print(v.name(), ' = ', v.solution_value(), b[(t, blood_type, age)])
            print("Total Cost =", solver.Objective().Value())

        return solver.Objective().Value(), {}


