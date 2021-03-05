import itertools
import random as rand
import copy
from typing import List, Tuple
from math import factorial as fact


class MonteCarlo:

    def __init__(self, k: int, m: int, n: int):
        self.k = k
        self.m = m
        self.n = n
        self.max_moves = n*m
        self.boards = {}
        self.all_positions = [(x, y) for x in range(m) for y in range(n)]
        self.root = self.make_root(m, n)
        self.term_states = {x: 0 for x in range(1, n*m + 1)}
        self.non_term_states = {x: 1 for x in range(1, n*m + 1)}
        self.illegal_states = {x:0 for x in range(1, n*m + 1)}
        self.avg_game_depth = 0
        self.games_played = 0
        self.states_per_turn = self.find_states_per_turn(n * m)
        self.non_term_estimate = 0

    def play_game(self):
        game = rand.sample(self.all_positions, len(self.all_positions))
        curr = self.root
        self.games_played += 1
        depth = 0
        while not (curr.terminal or depth > self.max_moves):
            depth += 1
            move = game.pop()
            new_pos = copy.deepcopy(curr.positions)
            new_player = "First" if curr.player == "Second" else "Second"
            new_pos[move[0]][move[1]] = "1" if curr.player == "First" else "2"
            board_key = "".join(itertools.chain.from_iterable(new_pos+[[new_player]]))
            if board_key in self.boards:
                curr = self.boards[board_key]
            else:
                terminal = self.check_terminal(curr.positions, move, curr.player) if game else True
                curr = BoardState(new_player, new_pos, terminal)
                self.boards[board_key] = curr
                if curr.terminal:
                    self.term_states[depth] += 1
                else:
                    self.non_term_states[depth] += 1
        self.avg_game_depth = (self.avg_game_depth * (self.games_played - 1) + depth) / self.games_played
        winner = "First" if curr.player == "Second" else "Second"
        while depth < self.max_moves:
            depth += 1
            game.pop()
            self.term_states[depth] += 1
            # write second check function


    def check_terminal(self, positions: List[List[str]], move: Tuple[int, int], player: str):
        target = "1" if player == "First" else "2"
        directions = [(1, 0, -1, 0), (0, 1, 0, -1), (1, 1, -1, -1), (-1, 1, -1, 1)]
        totals = []
        for direction in directions:
            total = 1
            forward = 1
            backward = 1
            coordinate = (forward*direction[0]+move[0], forward*direction[1]+move[1])
            while 0 <= coordinate[0] < self.m and 0 <= coordinate[1] < self.n and positions[coordinate[0]][coordinate[1]] == target:
                total += 1
                forward += 1
                coordinate = (forward * direction[0] + move[0], forward * direction[1] + move[1])
            coordinate = (backward*direction[2]+move[0], backward*direction[3]+move[1])
            while 0 <= coordinate[0] < self.m and 0 <= coordinate[1] < self.n and positions[coordinate[0]][coordinate[1]] == target:
                total += 1
                backward += 1
                coordinate = (backward * direction[2] + move[0], backward * direction[3] + move[1])
            totals.append(total)
        return True if max(totals) >= self.k else False

    def make_root(self, m: int, n: int):
        positions = [["0" for _ in range(n)] for row in range(m)]
        root = BoardState("First", positions)
        self.boards[("".join(itertools.chain.from_iterable(positions+[["First"]])))] = root
        return root

    def find_states_per_turn(self, positions: int):
        totals = {x: 0 for x in range(1, positions+1)}
        for turn in range(1, positions+1):
            m = turn//2
            n = turn - m
            totals[turn] = fact(positions) / (fact(n)*fact(m)*fact(positions-turn))
        return totals

    def update_non_term_estimate(self):
        proportions = estimate_proportions(self)
        self.non_term_estimate = 0
        for turn in range(1, self.max_moves+1):
            self.non_term_estimate += proportions[turn-1] * self.states_per_turn[turn]

    def simulate_n_games(self, n: int):
        for i in range(n):
            self.play_game()
            if not i%500:
                print(len(self.boards), self.avg_game_depth)
                print(self.non_term_states)
                print(self.term_states)
        self.update_non_term_estimate()
        print(self.non_term_estimate)



class BoardState:
    def __init__(self, player: str, positions: List[List[str]], terminal=False):
        self.player = player
        self.positions = positions
        self.terminal = terminal


def estimate_proportions(mc_record: 'MonteCarlo'):
    """

    :param mc_record:
    :return:
    Uses the proportion of non-terminal states at each step to estimate how much larger the search space in the next
    step is by multiplying that proportion by the number of possible choices. It assumes that no one can win on the
    first move
    """
    total = 1
    proportions = [1]
    for turn in range(1, mc_record.max_moves):
        non_term = mc_record.non_term_states[turn]
        term = mc_record.term_states[turn]
        prop_nt = non_term/(term+non_term)
        # choices_nt = mc_record.max_moves - turn
        # next_turn_states = proportions[-1]*prop_nt*choices_nt
        # total += next_turn_states
        # proportions.append(next_turn_states)
        proportions.append(prop_nt)
    return proportions






a = MonteCarlo(4, 4, 9)
a.simulate_n_games(500000)
print(a.states_per_turn)