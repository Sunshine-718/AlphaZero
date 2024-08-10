class Player:
    def __init__(self):
        self.win_rate = float('nan')
        self.mcts = None

    def reset_player(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def get_action(self, *args, **kwargs):
        raise NotImplementedError
