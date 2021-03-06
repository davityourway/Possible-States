import itertools
import random as rand
from typing import List, Tuple
from math import factorial


class MonteCarlo:

    def __init__(self, k: int, m: int, n: int):
        """
        MonteCarlo stores the results of the simulation and sampling on the object while keeping a dictionary of states
        encoded in strings to prevent double counting

        :param k: k in a row, column, or diagonal to reach a terminal state
        :param m: number of rows
        :param n: number of columns
        """
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
        """
        plays a full game of and then continues sampling to ensure an unbiased sample of possible board states at
        each timestep. We currently treat terminal states and illegal states the same way
        when incrementing the terminal state dictionary :return:
        """
        game = rand.sample(self.all_positions, len(self.all_positions))
        curr = self.make_root(self.m, self.n)
        self.games_played += 1
        depth = 0
        while depth < self.max_moves:
            depth += 1
            move = game.pop()
            next_player = "First" if curr.player == "Second" else "Second"
            curr.positions[move[0]][move[1]] = "1" if curr.player == "First" else "2"
            board_key = "".join(itertools.chain.from_iterable(curr.positions))
            if board_key in self.boards:
                curr.terminal = self.boards[board_key]
            else:
                terminal = self.check_terminal(curr.positions, move, curr.player) if not curr.terminal else True
                curr.terminal = terminal
                self.boards[board_key] = curr.terminal
            curr.player = next_player
            if curr.terminal:
                self.term_states[depth] += 1
            else:
                self.non_term_states[depth] += 1

    def check_terminal(self, positions: List[List[str]], move: Tuple[int, int], player: str):
        """
        checks if a state is terminal
        :param positions: matrix of positions
        :param move: coordinate of move to be made next
        :param player: 1st or 2nd player
        :return: returns whether or not there is a sequence greater than or equal to k that comes from making the move
        """
        target = "1" if player == "First" else "2"
        directions = [(1, 0, -1, 0), (0, 1, 0, -1), (1, 1, -1, -1), (-1, 1, 1, -1)]
        totals = []
        for direction in directions:
            total = 1
            total += self.k_in_direction(move, (direction[0], direction[1]), positions, target)
            total += self.k_in_direction(move, (direction[2], direction[3]), positions, target)
            totals.append(total)
        return True if max(totals) >= self.k else False

    def k_in_direction(self, start: Tuple[int, int], direction: Tuple[int, int], positions: List[List[str]], target: str):
        """
        returns the number of a target letter in a row
        :param start: non inclusive beginning of the search
        :param direction: tuple defining direction of the search
        :param positions: matrix representing the board state
        :param target: the variable in a row we are looking for
        :return: total number in a row in stipulated direction
        """
        increment = 1
        total = 0
        coordinate = (direction[0]*increment + start[0], direction[1]*increment + start[1])
        while 0 <= coordinate[0] < self.m and 0 <= coordinate[1] < self.n and positions[coordinate[0]][coordinate[1]] == target:
            total += 1
            increment += 1
            coordinate = (direction[0] * increment + start[0], direction[1] * increment + start[1])
        return total

    def make_root(self, m: int, n: int):
        """
        Generates the base state of the game
        :param m: rows
        :param n: columns
        :return: BoardState object with empty board
        """
        positions = [["0" for _ in range(n)] for row in range(m)]
        root = BoardState("First", positions, False)
        self.boards[("".join(itertools.chain.from_iterable(positions)))] = root
        return root

    def find_states_per_turn(self, positions: int):
        """
        determines how many possible states (of all kinds, even illegal) can come from each turn
        using the forumla from https://psyarxiv.com/rhq5j pg.19 of Supplemental Material

        :param positions: total number of positions that can be filled
        :return: a list of the total number of states (terminal, non-terminal, and illegal) for each turn
        """
        totals = {x: 0 for x in range(1, positions+1)}
        for turn in range(1, positions+1):
            m = turn//2
            n = turn - m
            totals[turn] = factorial(positions) / (factorial(n) * factorial(m) * factorial(positions - turn))
        return totals

    def update_non_term_estimate(self):
        """
        calculates an estimate of the total number of non-terminal states, updates the object parameter
        :return:
        """
        proportions = estimate_proportions(self)
        self.non_term_estimate = 0
        for turn in range(1, self.max_moves+1):
            self.non_term_estimate += proportions[turn-1] * self.states_per_turn[turn]

    def simulate_n_games(self, n: int):
        """
        runs play_game() n times and prints results every 500 iterations. Updates the MC estimate of non terminal
        states at the end
        :param n:
        :return:
        """
        for i in range(n):
            self.play_game()
            if not i % 5000:
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


def estimate_proportions(mc_record: MonteCarlo):
    """

    :param mc_record:
    :return: array of the proportion of states at each time step
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
        proportions.append(prop_nt)
        """
        These are leftovers from an estimation of state size that I was going to try to extrapolate out. I'm going to
        leave it for now but it should probably just be tossed
        """
        # choices_nt = mc_record.max_moves - turn
        # next_turn_states = proportions[-1]*prop_nt*choices_nt
        # total += next_turn_states
        # proportions.append(next_turn_states)
    return proportions






a = MonteCarlo(4, 4, 9)
a.simulate_n_games(100000)
print(a.states_per_turn)
print(sum([a.states_per_turn[x] for x in range(1,10)]))
