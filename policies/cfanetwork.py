from simulator.state import State
from ortools.linear_solver import pywraplp
from policies.policy import Policy


class CFANetwork(Policy):
    """
    Policy based on Value Function Approximation (VFA) network
    but based on storage_value parameter
    """
    def __init__(self, args):
        self.name = 'cfa-with-arcs'
        self.require_training = False
        # Create the mip mip_solver with the CBC backend.
        self.mip_solver = pywraplp.Solver('simple_mip_program',
                                          pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        self.storage_value = 10

    def get_actions(self, state: State, reward_map: dict(),
                    allowed_blood_transfers: dict()):
        """
        supply_attributes: blood supply attributes (type, age)
        demand_attributes: blood demand attributes (type, surgery, substitution)
        """
        # Decision (blood_type, age, (blood_type, surgery, substitution))
        solution = self.solve(state.supply, state.demands, reward_map,
                              allowed_blood_transfers,
                              print_solution=False, print_model=False)
        return solution['actions']

    def solve(self, supply: dict, demand: dict, reward_map: dict,
              allowed_blood_transfers,
              print_solution=False, print_model=False):
        # def get_network()
        max_blood_age = max(age for _, age in supply.keys()) + 1
        supply_demand_arcs = [(s, d) for s in supply.keys() for d in demand.keys()
                              if allowed_blood_transfers[s[0], d[0]] and (d[2] or s[0] == d[0])]
        supply_nextsupply_arcs = [(s, ("f", s[0], s[1] + 1)) for s in supply]
        nextsupply_sink_arcs = [(("f", s[0], s[1] + 1), (s[0], s[1] + 1, i))
                                for s in supply
                                for i in range(1, supply[s] + 1)]
        supplysink_sink = [((s[0], s[1] + 1, i), "sink") for s in supply for i in range(1, supply[s] + 1)]
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
            **{arc: self.storage_value * (max_blood_age - arc[0][1]) for arc in supplysink_sink}
        }

        # Clear model
        self.mip_solver.Clear()

        # A dictionary vars created to contain the referenced variables
        x = {(head, tail): self.mip_solver.NumVar(lb=0, ub=upper.get((head, tail), self.mip_solver.Infinity()),
                                                  name=f'x_{head}_{tail}') for head, tail in arcs}

        # Balance constraint
        for i in nodes:
            self.mip_solver.Add(
                sum(x[i, j] for j in outer[i]) - sum(x[j, i] for j in inner[i]) == b.get(i, 0)
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
        solution = {'actions': {(arc[0], arc[1]): round(x[arc].solution_value())
                                for arc in supply_demand_arcs
                                if x[arc].solution_value() >= 0.999},
                    'cost': (self.mip_solver.Objective().Value()
                             if solver_status == pywraplp.Solver.OPTIMAL else None),
                    'status': solver_status}

        return solution
