import numpy as np
from numpy.random import RandomState


class Scenario(object):
    def __init__(self, index, rnd_generator: RandomState, params):
        self.index = index
        self.rnd_generator = rnd_generator
        self.epochs = params['epochs']

        self.blood_types = params["blood_types"]
        self.max_blood_age = params['max_age']
        self.blood_ages = list(range(self.max_blood_age))
        self.donation_means = params["donation_means"]
        self.demand_means = params["donation_means"]

        self.blood_transfers = params["blood_transfers"]
        self.transfer_rewards = params["transfer_rewards"]
        self.time_periods_surge = params["time_periods_surge"]
        self.surge_prob = params["surge_prob"]
        self.surge_factor = params['surge_factor']
        self.surgery_types = params['surgery_types']
        self.surgery_types_prop = params['surgery_types_prop']
        self.substitution = params["substitution"]
        self.substitution_prop = params['substitution_prop']
        self.demand_types = [(i, j, k) for i in self.blood_types for j in self.surgery_types for k in self.substitution]

        self.demands = self.generate_demands(self.epochs)
        self.donations = self.generate_donations(self.epochs)

        self.blood_groups = [(i, j) for i in self.blood_types for j in self.blood_ages]
        self.init_blood_inventory = self.generate_init_blood_inventory()

        self.reward_map = self.get_epoch_reward_map()
        self.perfect_solution_reward = None

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
        return [{i: int(self.rnd_generator.poisson(self.donation_means[i])) for i in self.blood_types} for _ in range(epochs)]

    def generate_init_blood_inventory(self):
        multiplier = {age: .9 if age == 0 else (0.1 / (self.max_blood_age - 1)) for age in range(self.max_blood_age)}
        return {(blood_type, age): int(self.rnd_generator.poisson(self.donation_means[blood_type]) * multiplier[age])
                for blood_type, age in self.blood_groups}

    def perfect_information_solution(self):
        pass

    def get_epoch_reward_map(self):
        # map keys = ((blood_type, age), (blood_type, surgery, substitution))
        reward_map = {(s, d): 0 for s in self.blood_groups for d in self.demand_types if s[0] == d[0] and d[2]}
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

