from copy import deepcopy


def is_valid_decision(post_state_values: dict):
    return all([value >= 0 for value in post_state_values.values()])


class State:
    def __init__(self, blood_inventory: dict, demands: dict):
        self.supply = blood_inventory
        self.demands = demands

    def post_decision_state(self, decisions):
        used_supply = {(blood_type, age): sum(decisions.get(((blood_type, age), d), 0) for d in self.demands.keys())
                       for blood_type, age in self.supply.keys()}
        unsatisfied_demand = {
            (blood_type, surgery, substitution):
                self.demands[(blood_type, surgery, substitution)]
                - sum(decisions.get((s, blood_type, surgery, substitution), 0) for s in self.supply.keys())
            for blood_type, surgery, substitution in self.demands.keys()
        }
        post_state = {(blood_type, age): (self.supply[(blood_type, age - 1)] - used_supply[(blood_type, age - 1)]
                                          if age > 0 else 0) for blood_type, age in self.supply.keys()}

        return post_state if is_valid_decision(post_state) and is_valid_decision(unsatisfied_demand) else None

    def transition(self, post_decision_state, next_donations, next_demands):
        next_supply = post_decision_state
        blood_types = set([blood_type for blood_type, _ in self.supply.keys()])
        for blood_type in blood_types:
            next_supply[(blood_type, 0)] = next_donations[blood_type]

        return State(blood_inventory=next_supply, demands=next_demands)
