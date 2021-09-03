from copy import deepcopy
from simulator.state import State
from ortools.linear_solver import pywraplp
from policies.policy import Policy


class CFA(Policy):
    def __init__(self, args):
        self.require_training = False
        # Create the mip mip_solver with the CBC backend.
        self.mip_solver = pywraplp.Solver('simple_mip_program',
                                          pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    def get_actions(self, state: State, reward_map: dict()):
        """
        supply_attributes: blood supply attributes (type, age)
        demand_attributes: blood demand attributes (type, surgery, substitution)
        """
        # Decision (blood_type, age, (blood_type, surgery, substitution))
        solution = self.solve(state.supply, state.demands, reward_map,
                              print_solution=False, print_model=False)
        return solution['actions']

    def solve(self, supply: dict, demand: dict, reward_map: dict,
              print_solution=False, print_model=False):
        # def get_network()
        supply_demand_arcs = [(s, d) for s in supply.keys() for d in demand.keys() if s[0] == d[0] and d[2]]
        supply_sink_arcs = [(s, "sink") for s in supply]
        demand_sink_arcs = [(d, "sink") for d in demand]
        arcs = supply_demand_arcs + supply_sink_arcs + demand_sink_arcs
        nodes = list(supply.keys()) + list(demand.keys()) + ['sink']

        # Adjacency dictionary
        inner = {node: set() for node in nodes}
        outer = {node: set() for node in nodes}
        for arc in arcs:
            outer[arc[0]].add(arc[1])
            inner[arc[1]].add(arc[0])

        b = {**{n: s for n, s in supply.items()},
             **{n: 0 for n, d in demand.items()},
             **{"sink": -1 * sum(supply.values())}}
        upper = {arc: demand[arc[0]] for arc in demand_sink_arcs}
        reward = {arc: reward_map[arc] for arc in supply_demand_arcs}  # ToDo: Compute arc costs

        # Clear model
        self.mip_solver.Clear()

        # A dictionary vars created to contain the referenced variables
        x = {(head, tail): self.mip_solver.NumVar(lb=0, ub=upper.get((head, tail), self.mip_solver.Infinity()),
                                                  name=f'x_{head}_{tail}') for head, tail in arcs}

        # Balance constraint
        for i in nodes:
            self.mip_solver.Add(
                sum(x[i, j] for j in outer[i]) - sum(x[j, i] for j in inner[i]) == b[i]
            )

        # Objective function
        reward_function = [x[arc] * (reward + 10 * arc[0][1]) for arc, reward in reward.items()]
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



def other_solver(self, print_model, print_solution):
        state = []
        day = state[0]
        families = [i for i, e in enumerate(state[self.flag:]) if e != 0]
        penalty = self.get_penalty(self.occupancy, state[1])


        # A dictionary vars created to contain the referenced variables
        x = {i: self.mip_solver.BoolVar(f'x_{i}') for i in families}
        y = {o: self.mip_solver.BoolVar(f'y_{o}') for o in self.occupancy}

        # Upper & Lower constraints
        people = [x[i] * self.members[i] for i in families]
        self.mip_solver.Add(sum(people) <= self.max_people, 'max_people')
        self.mip_solver.Add(sum(people) >= self.min_people, 'min_people')

        # Occupancy constraint
        tmp = [y[o] * o for o in self.occupancy]
        self.mip_solver.Add(sum(people) == sum(tmp), 'occupancy_val')
        self.mip_solver.Add(sum([y[o] for o in self.occupancy]) == 1, 'trunc_occupancy')

        # Future constraints (necessary?)
        unassigned_people = [(1 - x[i]) * self.members[i] for i in families]
        self.mip_solver.Add(sum(unassigned_people) <= self.max_people * day, 'max_future')
        self.mip_solver.Add(sum(unassigned_people) >= self.min_people * day, 'min_future')

        # Objective function
        myopic_expr = [x[i] * self.costs[(day, i)] for i in families]
        occupancy_expr = [y[o] * penalty[i] for i, o in enumerate(self.occupancy)]
        future_assign_expr = [(1-x[i]) * self.v.get((day - 1, i), 0) for i in families]
        future_occupancy_expr = [y[o] * self.v_.get((day - 1, o), 0) for o in self.occupancy]
        self.mip_solver.Minimize(self.mip_solver.Sum(myopic_expr) + self.mip_solver.Sum(occupancy_expr) +
                                 self.mip_solver.Sum(future_assign_expr) + self.mip_solver.Sum(future_occupancy_expr))

        if print_model:
            print(self.mip_solver.ExportModelAsLpFormat(False))

        status = self.mip_solver.Solve()

        if status == pywraplp.self.mip_solver.OPTIMAL and print_solution:
            for name, v in x.items():
                if v.solution_value() >= 0.999:
                    print(v.name(), ' = ', v.solution_value())
            for name, v in y.items():
                if v.solution_value() >= 0.999:
                    print(v.name(), ' = ', v.solution_value())
            print("Total Cost =", self.mip_solver.Objective().Value())

        optimal = (status == pywraplp.Solver.OPTIMAL)
        solution = {'x': [i for i, v in x.items() if v.solution_value() >= 0.999],
                    'people': sum([i for i, v in y.items() if v.solution_value() >= 0.999]),
                    'cost': self.mip_solver.Objective().Value() if status == pywraplp.Solver.OPTIMAL else
                    self.infactibility_penalization}

        return solution, optimal, not optimal or len(solution['x']) == len(families)
