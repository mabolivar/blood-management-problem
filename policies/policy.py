from simulator.state import State


class Policy:
    def __init__(self, args):
        self.require_training = False

    def get_actions(self, state: State, reward_map: dict()):
        pass
