from simulator.scenario import Scenario
class State:
    def __init__(self, blood_inventory, demands):
        self.blood_inventory = blood_inventory
        self.next_demand = demands