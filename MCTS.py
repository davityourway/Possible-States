import itertools
import random as rand
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import List, Tuple
from math import factorial as fact


class MonteCarlo:
    """


    """
    def __init__(self, k: int, m: int, n: int):
        self.k = k
        self.m = m
        self.n = n
        self.boards = {}
        self.all_positions = [(x, y) for x in range(m) for y in range(n)]
        self.root = self.make_root(m, n)
        self.term_states = {x: 0 for x in range(n*m)}
        self.non_term_states = {x: 0 for x in range(n*m)}
        self.avg_depth = 0
        self.games_played = 0
        self.possible_games = self.total_per_tier(n*m)
        self.term_estimate = 0

    def run_n_times(self, ):

    def play_game(self):
        game = rand.sample(self.all_positions, len(self.all_positions))
        curr = self.root
        self.games_played += 1
        depth = 0
        while not curr.terminal:
            depth += 1
            move = game.pop()
            new_pos = curr.positions[:][:]
            new_player = "First" if curr.player == "Second" else "Second"
            new_pos[move[0]][move[1]] = "1" if curr.player == "First" else "2"
            board_key = "".join(itertools.chain.from_iterable(new_pos+[[new_player]]))
            if board_key in self.boards:
                curr = self.boards[board_key]
            else:
                terminal = self.check_move(curr.positions, move, curr.player) if game else True
                curr = BoardState(new_player, new_pos, terminal)
                self.boards[board_key] = curr
                if curr.terminal:
                    self.term_states[depth] += 1
                else:
                    self.non_term_states[depth] += 1
        self.avg_depth += depth/self.games_played

    def check_move(self, positions: List[List[str]], move: Tuple[int,int], player: str):
        target = "1" if player == "First" else "2"
        directions = [(1, 0, -1, 0), (0, 1, 0, -1), (1, 1, -1, -1), (-1, -1, -1, 1)]
        totals = []
        for direction in directions:
            total = 1
            forward = 1
            backward = 1
            coordinate = (forward*direction[0]+move[0], forward*direction[1]+move[1])
            while coordinate[0] < self.n and 0 <= coordinate[1] < self.m and positions[coordinate[0]][coordinate[1]] == target:
                total += 1
                forward += 1
            coordinate = (backward*direction[2]+move[0], backward*direction[3]+move[1])
            while 0 <= coordinate[0] and 0 <= coordinate[1] < self.m and positions[coordinate[0]][coordinate[1]] == target:
                total += 1
                backward += 1
        return True if max(totals) >= self.k else False

    def make_root(self, m: int, n: int):
        positions = [["0" for _ in range(n)] for row in range(m)]
        root = BoardState("First", positions)
        self.boards[("".join(itertools.chain.from_iterable(positions+[["First"]])))] = root
        return root

    def total_per_tier(self, positions: int):
        totals = {x: 0 for x in range(1, positions+1)}
        for turn in range(positions):
            m = turn//2
            n = turn - m
            totals[turn] = fact(positions) / (fact(n)*fact(m)*fact(positions-(n+m)))
        return totals

    def update_term_estimate(self):
        self.term_estimate = sum([self.term_states[x] / (self.term_states[x] + self.non_term_states[x]) for x in range(self.n * self.m)])


class BoardState:
    """
    """
    def __init__(self, player: str, positions: List[List[str]], terminal=False):
        self.player = player
        self.positions = positions
        self.terminal = terminal


a = MonteCarlo(2, 2, 2)
print(a.boards)
print(rand.sample(a.root.open_pos, len(a.root.open_pos)))