import copy
from math import ceil
from simulator.state import State
from ortools.linear_solver import pywraplp
from policies.policy import Policy


def single_update_algorithm(V, prev_epoch, slopes, alpha, delta):
    """ Updates the given slopes """
    for node, node_slopes in slopes.items():
        blood_type = node[0]
        prev_age = node[1] - 1
        node_slopes.sort(key=lambda x: -1 * x["supply"])
        for values in node_slopes:
            unit = values["supply"]
            prev_V = V[prev_epoch][(blood_type, prev_age)].get(unit, 0)
            V[prev_epoch][(blood_type, prev_age)][unit] = (1 - alpha) * prev_V + alpha * values['slope']


def leveling_algorithm(V, prev_epoch, slopes, alpha, delta):
    """ Levels up the slope functions to maintain monotonicity
        based on latest updated slopes """
    for node, node_slopes in slopes.items():
        blood_type = node[0]
        prev_age = node[1] - 1
        if prev_age < 0:
            continue
        node_slopes.sort(key=lambda x: 1 * x["supply"])

        for values in node_slopes:
            unit = values["supply"]
            prev_V = V[prev_epoch][(blood_type, prev_age)].get(unit, 0)
            V[prev_epoch][(blood_type, prev_age)][unit] = (1 - alpha) * prev_V + alpha * values['slope']

        V_update = V[prev_epoch][(blood_type, prev_age)]
        prev_sloped_unit = 0
        for values in node_slopes:
            units = values["supply"]
            v_ref = V_update[units]
            lower_range = range(prev_sloped_unit + 1, units + 1)
            upper_range = range(units + 1, units + delta + 1)
            for unit in lower_range[::-1]:
                current_V = V_update.get(unit, 0)
                next_V = V_update.get(unit + 1, v_ref)
                V_update[unit] = max((1 - alpha) * current_V + alpha * values["slope"], next_V)

            for unit in upper_range:
                current_V = V_update.get(unit, v_ref)
                prev_V = V_update.get(unit - 1, v_ref)
                V_update[unit] = min((1 - alpha) * current_V + alpha * values["slope"], prev_V)

            prev_sloped_unit = units


def cave_algorithm(V, prev_epoch, slopes, alpha, delta):
    """ CAVE algorithm - Use input slopes to update nearby
        supply values based on the delta param"""
    pass


class VFA(Policy):
    def __init__(self, args):
        self.name = 'vfa'
        self.require_training = False
        # Create the mip mip_solver with the CBC backend.
        self.mip_solver = pywraplp.Solver('simple_mip_program',
                                          pywraplp.Solver.CLP_LINEAR_PROGRAMMING)

        self.total_iterations = args["test_simulations"]
        self.num_iteration = 0
        self.V = {
            t: {
                (b, age): {0: 0}
                for b in args["blood_types"]
                for age in range(args["max_age"])
            } for t in range(args["epochs"])
        }
        self.V_history = {
            t: {
                (b, age): dict()
                for b in args["blood_types"]
                for age in range(args["max_age"])
            } for t in range(args["epochs"])
        }

    def get_actions(self, state: State, reward_map: dict(),
                    allowed_blood_transfers: dict()):
        """
        supply_attributes: blood supply attributes (type, age)
        demand_attributes: blood demand attributes (type, surgery, substitution)
        """
        # Decision (blood_type, age, (blood_type, surgery, substitution))
        solution = self.solve(state.epoch, state.supply, state.demands, reward_map,
                              allowed_blood_transfers,
                              print_solution=False, print_model=False)
        if state.epoch > 0:
            slopes = self.compute_slopes(solution, state.epoch, state.supply, state.demands, reward_map,
                                     allowed_blood_transfers)

            self.update_value_estimates(
                prev_epoch=state.epoch - 1,
                slopes=slopes
            )
        return solution['actions']

    def compute_slopes(self, base_solution, epoch, base_supply, base_demands, reward_map, allowed_blood_transfers):
        slope_steps = 1
        supply = copy.deepcopy(base_supply)
        slopes = {node: [] for node in supply.keys()}
        for node in supply.keys():
            if node[1] == 0:
                continue

            current_cost = base_solution["cost"]
            # Compute positive slopes
            for i in range(slope_steps):
                supply[node] += 1
                new_cost = self.solve(epoch, supply, base_demands, reward_map,
                                 allowed_blood_transfers)["cost"]
                slope = new_cost - current_cost
                slopes[node].append({"slope": slope, "supply": supply[node]})
                if slope == 0:
                    break
                current_cost = new_cost

            # Compute negative slopes
            supply[node] = base_supply[node]
            current_cost = base_solution["cost"]
            for i in range(slope_steps):
                supply[node] -= 1
                if supply[node] < 0:
                    break
                new_cost = self.solve(epoch, supply, base_demands, reward_map,
                                 allowed_blood_transfers)["cost"]
                slope = current_cost - new_cost
                slopes[node].append({"slope": slope, "supply": supply[node] + 1})
                current_cost = new_cost
            supply[node] = base_supply[node]

        return slopes

    def update_value_estimates(self, prev_epoch: int, slopes: dict):
        alpha = 0.8 * (1 - (self.num_iteration + 1) / self.total_iterations)
        delta = ceil(10 * (1 - (self.num_iteration + 1) / self.total_iterations))
        if prev_epoch < 0:
            return
        # Value function update algorithm
        # as alternative: single_update_algorithm() or cave_algorithm()
        leveling_algorithm(self.V, prev_epoch, slopes, alpha, delta)

    def solve(self, epoch: int, supply: dict, demand: dict, reward_map: dict,
              allowed_blood_transfers,
              print_solution=False, print_model=False):
        max_blood_age = max(list(zip(*supply.keys()))[1])
        # def get_network()
        supply_demand_arcs = [(s, d) for s in supply.keys() for d in demand.keys()
                              if allowed_blood_transfers[s[0], d[0]] and (d[2] or s[0] == d[0])]
        supply_nextsupply_arcs = [(s, ("f", s[0], s[1])) for s in supply]
        nextsupply_sink_arcs = [(("f", s[0], s[1]), (s[0], s[1], i))
                                for s in supply
                                for i in range(1, supply[s] + 1)]
        supplysink_sink = [((s[0], s[1], i), "sink") for s in supply for i in range(1, supply[s] + 1)]
        demand_sink_arcs = [(d, "sink") for d in demand]
        arcs = supply_demand_arcs + supply_nextsupply_arcs + nextsupply_sink_arcs + supplysink_sink + demand_sink_arcs

        head_nodes, tail_nodes = [set(nodes) for nodes in zip(*arcs)]
        nodes = set.union(head_nodes, tail_nodes)

        # Adjacency dictionary
        inner = {node: set() for node in nodes}
        outer = {node: set() for node in nodes}
        for arc in arcs:
            outer[arc[0]].add(arc[1])
            inner[arc[1]].add(arc[0])

        b = {**{n: s for n, s in supply.items()},
             **{n: 0 for n, d in demand.items()},
             **{"sink": -1 * sum(supply.values())}}
        upper = {
            **{arc: demand[arc[0]] for arc in demand_sink_arcs},
            **{arc: 1 for arc in supplysink_sink}
        }
        reward = {
            **{arc: reward_map[arc] for arc in supply_demand_arcs},
            **{arc: 0 if arc[0][1] > max_blood_age else self.V[epoch][(arc[0][0], arc[0][1])].get(arc[0][2], 0)
               for arc in supplysink_sink}
        }

        # Clear model
        self.mip_solver.Clear()

        # A dictionary vars created to contain the referenced variables
        x = {(head, tail): self.mip_solver.NumVar(lb=0, ub=upper.get((head, tail), self.mip_solver.Infinity()),
                                                  name=f'x_{head}_{tail}') for head, tail in arcs}

        # Balance constraint
        for i in nodes:
            self.mip_solver.Add(
                sum(x[i, j] for j in outer[i]) - sum(x[j, i] for j in inner[i]) == b.get(i, 0),
                name=str(i)
            )

        # Objective function
        reward_function = [x[arc] * reward for arc, reward in reward.items()]
        self.mip_solver.Maximize(self.mip_solver.Sum(reward_function))

        if print_model:
            print(self.mip_solver.ExportModelAsLpFormat(False))

        solver_status = self.mip_solver.Solve()

        if solver_status == pywraplp.Solver.OPTIMAL and print_solution:
            for name, v in x.items():
                if v.solution_value() >= 0.999:
                    print(v.name(), ' = ', v.solution_value())
            print("Total Cost =", self.mip_solver.Objective().Value())

        optimal = (solver_status == pywraplp.Solver.OPTIMAL)
        solution = {
            'actions': {(arc[0], arc[1]): round(x[arc].solution_value())
                        for arc in supply_demand_arcs
                        if x[arc].solution_value() >= 0.999},
            'cost': (self.mip_solver.Objective().Value()
                     if solver_status == pywraplp.Solver.OPTIMAL else None),
            'status': solver_status,
            'slopes': {
                node: [{'slope': self.mip_solver.LookupConstraint(str(node)).DualValue(),
                       "supply": b}]
                for node, b in supply.items() if node[1] > 0
            }
        }

        return solution
