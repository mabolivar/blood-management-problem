import pandas as pd
from ortools.linear_solver import pywraplp

def create_network(supply, demand, allowed_blood_transfers, reward_map, V=dict()):
  """ Crete nodes/arcs graph to be optimized """
  # Create network
  supply_demand_arcs = [(s, d) for s in supply.keys() for d in demand.keys()
                        if (s, d) in allowed_blood_transfers]
  supply_nextsupply_arcs = [(s, ("f", s[0], s[1])) for s in supply]
  nextsupply_sink_arcs = [(("f", s[0], s[1]), (s[0], s[1], i))
                          for s in supply
                          for i in range(1, supply[s] + 1)]
  supplysink_sink = [((s[0], s[1], i), "sink") for s in supply for i in range(1, supply[s] + 1)]
  demand_sink_arcs = [(d, "sink") for d in demand]
  
  
  arcs = supply_demand_arcs + supply_nextsupply_arcs + nextsupply_sink_arcs + supplysink_sink + demand_sink_arcs
  
  head_nodes, tail_nodes = [set(nodes) for nodes in zip(*arcs)]
  nodes = set.union(head_nodes, tail_nodes)
  
  b = {**{n: s for n, s in supply.items()},
       **{n: 0 for n, d in demand.items()},
       **{"sink": -1 * sum(supply.values())}}
  upper = {
      **{arc: demand[arc[0]] for arc in demand_sink_arcs},
      **{arc: 1 for arc in supplysink_sink}
  }
  reward = {
      **{arc: reward_map[arc] for arc in supply_demand_arcs},
      **{arc: 0 for arc in supply_nextsupply_arcs + nextsupply_sink_arcs},
      **{arc: V.get((arc[0][0], arc[0][1]), dict()).get(arc[0][2], 0)
         for arc in supplysink_sink}
  }
  
  return nodes, b, arcs, upper, reward


def solve_network(nodes, b, arcs, upper, reward, print_model=False, print_solution=False):
  """ Optimize provided network """
  # Adjacency dictionary
  inner = {node: set() for node in nodes}
  outer = {node: set() for node in nodes}
  for arc in arcs:
      outer[arc[0]].add(arc[1])
      inner[arc[1]].add(arc[0])
  
  # Clear model
  mip_solver = pywraplp.Solver('simple_mip_program',
                                            pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
  
  # A dictionary vars created to contain the referenced variables
  x = {(head, tail): mip_solver.NumVar(lb=0, ub=upper.get((head, tail), mip_solver.Infinity()),
                                            name=f'x_{head}_{tail}') for head, tail in arcs}
  
  # Balance constraint
  for i in nodes:
      mip_solver.Add(
          sum(x[i, j] for j in outer[i]) - sum(x[j, i] for j in inner[i]) == b.get(i, 0),
          name=str(i)
      )
  
  # Objective function
  reward_function = [x[arc] * reward for arc, reward in reward.items()]
  mip_solver.Maximize(mip_solver.Sum(reward_function))
  
  if print_model:
      print(mip_solver.ExportModelAsLpFormat(False))
  
  solver_status = mip_solver.Solve()
  
  if solver_status == pywraplp.Solver.OPTIMAL and print_solution:
      for name, v in x.items():
          if v.solution_value() >= 0.999:
              print(v.name(), ' = ', v.solution_value())
      print("Total Cost =", mip_solver.Objective().Value())
  
  optimal = (solver_status == pywraplp.Solver.OPTIMAL)
  solution = {
      'actions': {(arc[0], arc[1]): round(x[arc].solution_value())
                  for arc in arcs
                  if x[arc].solution_value() >= 0.999},
      'cost': (mip_solver.Objective().Value()
               if solver_status == pywraplp.Solver.OPTIMAL else None),
      'status': solver_status,
      'duals': {
          node: {'dual': mip_solver.LookupConstraint(str(node)).DualValue(),
                 "supply": value}
          for node, value in b.items()
      }
  }
  return solution


def optimization_step(supply, demand, allowed_blood_transfers, 
    reward_map, V=dict()):
  """ Combine create_network(), solve_network() and generate duals processes """
  # Get network
  nodes, b, arcs, upper, reward = create_network(
    supply, 
    demand, 
    allowed_blood_transfers, 
    reward_map,
    V = V
    )
    
  # Solve network
  base_solution = solve_network(nodes, b, arcs, upper, reward)
  
  # Get duals
  duals = dict()
  for k, v in b.items():
    if k[0] == "s" and k != "sink":
      b[k]= v + 1
      b['sink']-= 1
      sol = solve_network(nodes, b, arcs, upper, reward)
      duals[k] = {"dual": sol["cost"] - base_solution["cost"], "supply": v + 1}
      b[k] = v
      b['sink']+= 1
    
  return base_solution, reward, duals


def transition_function(_inventory, _donations, 
                        _demand, _allowed_blood_transfer,
                        _reward_map):
  """ Update supply and demand values for next epoch """

  supply = {
    tuple(k.split("_")): v + _inventory.get(tuple(k.split("_")), 0)
    for k, v in _donations.items()
    }
  demand = {tuple(k.split("_")): v for k, v in _demand.items()}
  
  allowed_blood_transfers = set(
  (tuple(tail.split("_")), tuple(head.split("_")))
  for tail, head in _allowed_blood_transfer
  )
  reward_map= {(tuple(k[0].split("_")), tuple(k[1].split("_"))): v 
                for k, v in
                _reward_map.items()}
  return supply, demand, allowed_blood_transfers, reward_map


def solution_to_df(solution, num_iteration=None):
  """ Get a solution from `solve_network()` function
    and returns a data frame"""
  epochs = solution.keys()
  actions_df = pd.concat(
    pd.DataFrame.from_dict(
    solution[t]["actions"], 
    orient = "index",
    columns=[ "flow"]
    )
    .reset_index()
    .rename(columns={"index":"arc"})
    .assign(epoch = t) for t in epochs
    )
  reward_df = pd.concat(
    pd.DataFrame.from_dict(
    solution[t]["rewards"], 
    orient = "index",
    columns=[ "reward"]
    )
    .reset_index()
    .rename(columns={"index":"arc"})
    .assign(epoch = t) for t in epochs
    )
  return (reward_df
          .merge(actions_df, how="left")
          .assign(iteration = num_iteration)
         )
