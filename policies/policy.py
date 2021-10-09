from simulator.state import State


class Policy:
    def __init__(self, args):
        self.name = ""
        self.require_training = False

    def get_actions(self, state: State, reward_map: dict(),
                    allowed_blood_transfers: dict()):
        pass
