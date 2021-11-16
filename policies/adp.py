import copy
from math import ceil
from simulator.state import State
from ortools.linear_solver import pywraplp
from policies.policy import Policy


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
                (b, age): {i: [0] for i in range(0, 30)}
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
        if state.epoch == 0:
            self.num_iteration += 1

        solution = self.solve(state.epoch, state.supply, state.demands, reward_map,
                              allowed_blood_transfers,
                              print_solution=False, print_model=False)
        self.update_value_estimates(
            prev_epoch=state.epoch - 1,
            dual_variables=solution['duals']
        )
        return solution['actions']

    def update_value_estimates(self, prev_epoch: int, dual_variables: dict):
        alpha = ceil(0.5 * (1 - (self.num_iteration + 1) / self.total_iterations))
        delta = ceil(10 * (1 - (self.num_iteration + 1) / self.total_iterations))
        if prev_epoch < 0:
            return
        for node, values in dual_variables.items():
            blood_type = node[0]
            prev_age = node[1] - 1
            units = values["supply"] + 1

            update_range = range(max(0, units - delta), units + delta + 1)
            for unit in update_range:
                prev_V = self.V[prev_epoch][(blood_type, prev_age)].get(unit, 0)
                self.V[prev_epoch][(blood_type, prev_age)][unit] = min(self.V[prev_epoch][(blood_type, prev_age)].get(unit - 1, 99999999) - 0.5, (1 - alpha) * prev_V + alpha * values['dual'])

                self.V_history[prev_epoch][(blood_type, prev_age)][unit].append((self.num_iteration, self.V[prev_epoch][(blood_type, prev_age)][unit]))

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
            'duals': {
                node: {'dual': self.mip_solver.LookupConstraint(str(node)).DualValue(),
                       "supply": b}
                for node, b in supply.items() if node[1] > 0
            }
        }

        return solution
