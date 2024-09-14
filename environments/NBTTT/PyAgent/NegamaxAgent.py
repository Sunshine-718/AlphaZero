#!/usr/bin/python3
from .utils import status, dynamic_params


class NegamaxAgent:
    def __init__(self, boards, heuristic, player=0, depth=None):
        """
        :param boards: nine board, shape(10, 10).
        :param heuristic: Heurictic Object created by Heuristic class.
        :param player: 0 or 1.
        """
        self.boards = boards
        self.heuristic = heuristic
        self.player = player
        self.dynamic_ = True if depth is None else False
        self.depth = depth

    def action(self, curr) -> int:
        """
        Function to choose an action given a state "self.boards"
        :param curr: index of current board.
        :return: optimal action in current state based on the heuristic estimate
        """
        if self.dynamic_:
            self.depth, _ = dynamic_params(self.boards, curr, self.depth, 9)
        return self._alphabeta(curr, self.depth, 9)[1]

    def _alphabeta(self, curr, depth, branch=9) -> (int | float, int):
        """
        Negamax algorithm with alpha-beta pruning.
        :param curr: index of current board.
        :param depth: maximum search depth.
        :param branch: maximum node to expand at each level of tree, default 9 (maximum number of action in one board).
        :return: [0]: value of root node.
                 [1]: optimal action in current state based on the heuristic estimate.
        """

        def negamax(boards, curr, depth, player, alpha, beta):
            if status(boards, curr)[0] or depth == 0:
                return self.heuristic.estimate(boards, curr, player), None
            valid = self.heuristic.sorted_action(boards, curr, branch, player)
            value_eval = -float('inf')
            best_action = None
            for action in valid:
                boards[curr, action] = player
                value_eval = max(value_eval, -negamax(boards, action, depth - 1, 1 - player, -beta, -alpha)[0])
                boards[curr, action] = 2
                if value_eval > alpha:
                    alpha = value_eval
                    best_action = action
                    if alpha >= beta:
                        return alpha, best_action
            return alpha, best_action

        return negamax(self.boards, curr, depth, self.player, -float('inf'), float('inf'))


if __name__ == '__main__':
    pass
