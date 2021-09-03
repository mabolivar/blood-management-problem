from copy import deepcopy
from simulator.state import State
from policies.policy import Policy


class Basic(Policy):
    def __init__(self, args):
        self.require_training = False

    def get_actions(self, state: State, reward_map: dict()):
        # supply_attributes : blood supply attributes (type, age)
        # demand_attributes: blood demand attributes (type, surgery, substitution)
        supply_attributes = [key for key, value in state.supply.items() if value > 0]
        demand_attributes = [key for key, value in state.demands.items() if value > 0]
        # Decision (blood_type, age, (blood_type, surgery, substitution))
        x = {(a, d): 0 for a in supply_attributes for d in demand_attributes}

        current_inventory = deepcopy(state.supply)
        supply_attributes.sort(key=lambda y: y[1], reverse=True)
        ages = sorted(set(a for _, a in supply_attributes), reverse=True)
        for d in demand_attributes:
            for age in ages:
                x[((d[0], age), d)] = min(current_inventory[d[0], age], state.demands[d])
                current_inventory[d[0], age] -= x[((d[0], age), d)]

        return {k: v for k, v in x.items() if v > 0}

