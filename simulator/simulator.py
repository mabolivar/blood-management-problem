import numpy as np
from numpy.random import RandomState


class Simulator(object):
    def __init__(self, index, rnd_generator: RandomState, **kwargs):
        self.index = index
        self.epochs = kwargs['epochs']

        self.blood_types = kwargs["blood_types"]
        self.donation_means = kwargs["donation_means"]
        self.demand_means = kwargs["donation_means"]

        self.time_periods_surge = kwargs["time_periods_surge"]
        self.surge_prob = kwargs["surge_prob"]
        self.surge_factor = kwargs['surge_factor']
        self.surgery_types = kwargs['surgery_types']
        self.surgery_types_prop = kwargs['surgery_types_prop']
        self.substitution = kwargs["substitution"]

        self.demand_types = [(i, j, k) for i in self.blood_types for j in self.surgery_types for k in self.substitution]

        self.demands = self.generate_demands(self.epochs)
        self.donations = self.generation_donations(self.epochs)

        self.perfect_solution_reward = None

    def generate_demands(self, t, params):

        # demand
        if t in self.time_periods_surge and np.random.uniform(0, 1) < self.surge_prob:
            factor = self.surge_factor
        else:
            factor = 1

        demand = [int(np.random.poisson(
            factor * self.demand_means[dmd[0]] * self.surgery_types_prop[dmd[1]] *
            params['SUBSTITUTION_PROP'][dmd[2]])) for dmd in self.demand_types]

        if False:
            demand = []
            for dmd in Bld_Net.demandnodes:
                if dmd[0] == "O-":
                    if dmd[1] == "Urgent":
                        demand.append(1)
                    else:
                        eleDem = max(0, int(np.random.poisson(factor * params['MAX_DEM_BY_BLOOD'][dmd[0]] - 1)) - 1)
                        demand.append(eleDem)

                else:
                    demand.append(int(np.random.poisson(
                        factor * params['MAX_DEM_BY_BLOOD'][dmd[0]] * params['SURGERYTYPES_PROP'][dmd[1]] *
                        params['SUBSTITUTION_PROP'][dmd[2]])))

        return Exog_Info(demand, donation)

    def generate_donations(self, epochs):
        return {i: int(self.rnd_generator.poisson(self.donation_means[i]), size=epochs) for i in self.blood_types}

    def perfect_information_solution(self):
        pass