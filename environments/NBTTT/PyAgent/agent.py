#!/usr/bin/python3
########################################################################################################################
"""
Algorithm: Negamax, AlphaBeta Pruning, Heuristic Search

We use Negamax algorithm to search the game tree, because the state space of Nine-board Tic Tac Toe is very large,
it is necessary to use a heuristic evaluation function to accelerate searching, with unlimited computational budget,
Negamax guarantees to find the optimal sequence of actions, but in the real world, with limited computational budget,
it is impossible to search the whole tree at every step, with the help of heuristic function,
we can find the local optimal solution.

We notice that, dynamicly change the 'depth of searching' and 'branch factor' is very helpful to decrease the searching
time without lower too much of the quality of decision. At the beginning of the game, it is unnecessary to use a large
serch depth, in contrast, a large branch factor is more important. At the intermediate stage of the game,
it is important that to increase the search depth rapidly, and shrink the branch factor a little bit,
because in the middlegame, 'far-sighted' is significant to let the agent occupy the key point quickly.
When the game near the end, the state space is reletively small, so we decide to use a larger serch depth
to find the optimal action.
"""
########################################################################################################################
import socket
import sys
import numpy as np
from NegamaxAgent import NegamaxAgent
from utils import dynamic_params
from Heuristic import Heuristic, table

boards = np.zeros((10, 10), dtype="int8")
curr = 0
depth = 4
branch = 9
heuristic = Heuristic(boards, table)
agent = NegamaxAgent(boards, heuristic)


def play():
    global depth, branch
    depth, branch = dynamic_params(boards, curr, depth, branch)
    n = agent.action(curr, depth, branch)
    place(curr, n, 1)
    return n


def place(board, num, player):
    global curr
    curr = num
    boards[board][num] = player


def parse(string):
    if "(" in string:
        command, args = string.split("(")
        args = args.split(")")[0]
        args = args.split(",")
    else:
        command, args = string, []
    if command == "second_move":
        place(int(args[0]), int(args[1]), 2)
        return play()
    elif command == "third_move":
        place(int(args[0]), int(args[1]), 1)
        place(curr, int(args[2]), 2)
        return play()
    elif command == "next_move":
        place(curr, int(args[0]), 2)
        return play()
    elif command == "win":
        print("Yay!! We win!! :)")
        return -1
    elif command == "loss":
        print("We lost :(")
        return -1

    return 0


# connect to socket
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(sys.argv[2])  # Usage: ./agent.py -p (port)

    s.connect(('localhost', port))
    while True:
        text = s.recv(1024).decode()
        if not text:
            continue
        for line in text.split("\n"):
            response = parse(line)
            if response == -1:
                s.close()
                return
            elif response > 0:
                s.sendall((str(response) + "\n").encode())


if __name__ == "__main__":
    main()
