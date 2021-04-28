import itertools
import random as rand
from typing import List, Tuple, Optional, Set
from math import factorial
from scipy.stats import sem



def find_legal_junctions(win_list: List[Tuple[int, int]], k: int):
    win_size = len(win_list)
    distance_from_end = win_size - k
    if win_size == k:
        return set(win_list)
    else:
        return set(win_list[distance_from_end:-distance_from_end])


def find_states_per_turn(positions: int):
    """
    determines how many possible states (of all kinds, even illegal) can come from each turn
    using the forumla from https://psyarxiv.com/rhq5j pg.19 of Supplemental Material

    :param positions: total number of positions that can be filled
    :return: a list of the total number of states (terminal, non-terminal, and illegal) for each turn
    """
    totals = {x: 0 for x in range(positions+1)}
    totals[0] = 1
    for turn in range(1, positions+1):
        m = turn//2
        n = turn - m
        totals[turn] = factorial(positions) // (factorial(n) * factorial(m) * factorial(positions - turn))
    return totals


class MonteCarlo:

    def __init__(self, m: int, n: int, k: int):
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
        self.term_states = {x: 0 for x in range(n*m + 1)}
        self.non_term_states = {x: 0 for x in range(n*m + 1)}
        self.illegal_states = {x: 0 for x in range(n*m + 1)}
        # self.avg_game_depth = 0
        self.samples_generated = 0
        self.states_per_turn = find_states_per_turn(n * m)
        self.non_term_estimate = 0
        self.term_estimate = 0
        self.illegal_estimate = 0
        self.non_term_states[0] = 1.0

    def play_game(self):
        """
        plays a full game until we reach a terminal state, after which we check each state for its legality. There are
        two types of illegal states. After a player wins, every subsequent move from the other player causes
        a state that could not have happened (i.e. where a win is on the board for X and it is X to move).
        A simple player check determines that. If a move by the winning player creates an illegal configuration
        which we find with check_legal, we pop the game stack and increment the illegal states dictionary by one for each
        subsequent turn
         :return:
        """
        game = rand.sample(self.all_positions, len(self.all_positions))
        curr = self.make_root(self.m, self.n)
        self.samples_generated += 1
        depth = 0
        while depth < self.max_moves:
            depth += 1
            move = game.pop()
            next_player = "First" if curr.player == "Second" else "Second"
            curr.positions[move[0]][move[1]] = "1" if curr.player == "First" else "2"
            if not curr.terminal:
                curr.terminal, win_list = self.check_terminal(curr.positions, move, curr.player)
                if curr.terminal:
                    winner = curr.player
                    self.term_states[depth] += 1
                    win_set = find_legal_junctions(win_list, self.k)
                    _, win_set = self.check_legal(curr.positions, move, win_list, win_set, curr.player)
                else:
                    self.non_term_states[depth] += 1
            else:
                curr.legal, win_set = self.check_legal(curr.positions, move, win_list, win_set, curr.player)
                if not curr.legal:
                    self.illegal_states[depth] += 1
                    while game:
                        depth += 1
                        game.pop()
                        self.illegal_states[depth] += 1
                elif curr.player == winner:
                    self.term_states[depth] += 1
                else:
                    self.illegal_states[depth] += 1
            curr.player = next_player

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
        for direction in directions:
            total = 1
            fcount, fchecked = self.target_in_direction(move, (direction[0], direction[1]), positions, target)
            bcount, bchecked = self.target_in_direction(move, (direction[2], direction[3]), positions, target)
            if total + fcount + bcount >= self.k:
                bchecked.reverse()
                return True, bchecked + [move] + fchecked
        return False, []

    def check_legal(self, positions: List[List[str]], move: Tuple[int, int], win_list: List, win_set: Set, player: str):
        """
        function checks the legality by searching for a winning combination that branches off the most recent sample. IF
        it find one it asks if that sample intersects the original win at a valid node or nodes. If it does, it sets the
        valid win set to that intersection and returns True and the intersection, otherwise it returns false. There are
        some earlier checks that curtail the search space using some simpler heuristics to improve speed
        :param positions:
        :param move:
        :param win_list:
        :param win_set:
        :param player:
        :return:
        """
        target = "1" if player == "First" else "2"
        direction_pairs = [(1, 0, -1, 0), (0, 1, 0, -1), (1, 1, -1, -1), (-1, 1, 1, -1)]
        for direction in direction_pairs:
            total = 1
            fcount, fchecked = self.target_in_direction(move, (direction[0], direction[1]), positions, target)
            bcount, bchecked = self.target_in_direction(move, (direction[2], direction[3]), positions, target)
            total += fcount + bcount
            if total >= self.k:
                win_sample = win_list[0]
                if total >= self.k*2 or target != positions[win_sample[0]][win_sample[1]]:
                    return False, set()
                bchecked.reverse()
                new_win_list = bchecked + [move] + fchecked
                new_win_junctions = find_legal_junctions(new_win_list, self.k)
                win_intersection = new_win_junctions.intersection(win_set)
                if not win_intersection:
                    return False, set()
                win_set = win_intersection
        return True, win_set

    def target_in_direction(self, start: Tuple[int, int], direction: Tuple[int, int], positions: List[List[str]],
                            target: str, win_move: Optional[Tuple[int, int]] = None):
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
        checked = []
        while 0 <= coordinate[0] < self.m and 0 <= coordinate[1] < self.n and positions[coordinate[0]][coordinate[1]] == target:
            checked.append(coordinate)
            total += 1
            increment += 1
            coordinate = (direction[0] * increment + start[0], direction[1] * increment + start[1])
        return total, checked

    def make_root(self, m: int, n: int):
        """
        Generates the base state of the game
        :param m: rows
        :param n: columns
        :return: BoardState object with empty board
        """
        positions = [["0" for _ in range(n)] for row in range(m)]
        root = BoardState("First", positions, False, True)
        self.boards[("".join(itertools.chain.from_iterable(positions)))] = root
        return root

    def update_non_term_estimate(self):
        """
        calculates an estimate of the total number of non-terminal states, updates the object parameter
        :return:
        """
        proportions = estimate_nonterm_proportions(self)
        self.non_term_estimate = 0
        for turn in range(1, self.max_moves+1):
            self.non_term_estimate += proportions[turn] * self.states_per_turn[turn]

    def update_term_estimate(self):
        """
        calculates an estimate of the total number of terminal states, updates the object parameter
        :return:
        """
        proportions = estimate_term_proportions(self)
        self.term_estimate = 0
        for turn in range(1, self.max_moves+1):
            self.term_estimate += proportions[turn] * self.states_per_turn[turn]

    def update_illegal_estimate(self):
        """
        calculates an estimate of the total number of illegal states, updates the object parameter
        :return:
        """
        proportions = estimate_illegal_proportions(self)
        self.illegal_estimate = 0
        for turn in range(1, self.max_moves+1):
            self.illegal_estimate += proportions[turn] * self.states_per_turn[turn]

    def simulate_n_games(self, n: int):
        """
        runs play_game() n times and prints results every 500 iterations. Updates the MC estimate of non terminal
        states at the end
        :param n:
        :return:
        """

        for i in range(n):
            self.play_game()
        self.update_non_term_estimate()
        self.update_term_estimate()
        self.update_illegal_estimate()


class BoardState:
    def __init__(self, player: str, positions: List[List[str]], terminal: Optional[bool] = False, legal: Optional[bool] = True):
        self.player = player
        self.positions = positions
        self.terminal = terminal
        self.legal = legal


def estimate_nonterm_proportions(mc_record: MonteCarlo):
    """

    :param mc_record:
    :return: array of the proportion of states at each time step
    Gives the proportion of non-terminal states at each step in an array, which can be used as a probability estimate
    """
    proportions = [1.0]
    for turn in range(1, mc_record.max_moves+1):
        non_term = mc_record.non_term_states[turn]
        term = mc_record.term_states[turn]
        illegal = mc_record.illegal_states[turn]
        prop_nt = non_term/(term+non_term+illegal)
        proportions.append(prop_nt)
    return proportions


def estimate_term_proportions(mc_record: MonteCarlo):
    """

    :param mc_record:
    :return: array of the proportion of states at each time step
    Gives the proportion of terminal states at each step in an array, which can be used as a probability estimate
    """
    proportions = [0.0]
    for turn in range(1, mc_record.max_moves+1):
        non_term = mc_record.non_term_states[turn]
        term = mc_record.term_states[turn]
        illegal = mc_record.illegal_states[turn]
        prop_nt = term/(term+non_term+illegal)
        proportions.append(prop_nt)
    return proportions


def estimate_illegal_proportions(mc_record: MonteCarlo):
    """

    :param mc_record:
    :return: array of the proportion of states at each time step
    Gives the proportion of illegal states at each step in an array, which can be used as a probability estimate
    """
    proportions = [0.0]
    for turn in range(1, mc_record.max_moves+1):
        non_term = mc_record.non_term_states[turn]
        term = mc_record.term_states[turn]
        illegal = mc_record.illegal_states[turn]
        prop_nt = illegal/(term+non_term+illegal)
        proportions.append(prop_nt)
    return proportions


if __name__ == '__main__':
    k = 3
    m = 2
    n = 3
    runs = 10
    samples = 1000000
    # #
    # for _ in range(2):
    #     a = MonteCarlo(m, n, k)
    #     a.simulate_n_games(10000)
    #     print(a.non_term_estimate)
    #     print(a.term_estimate)
    #     print(a.illegal_estimate)
    #
    #     print(_)


    nonterm_means = []
    term_means = []
    illegal_means = []
    nonterm_sterrors = []
    term_sterrors = []
    illegal_sterrors = []
    #
    # for m in range(5, n+1):
    #     for k in range(3, m+1):
    #         print(m, n, k)
    #         if k == m == n:
#             break
    with open(f'k={k}m={m}n={n}.txt', 'a') as f:
# with open(f'{m}x{n}k{k}results.txt', 'a') as f:
        f.write(f"\n\n\nTest for {m}x{n} k={k}: \n")
        f.write(f"{runs} runs and {samples} samples \n\n")
        non_term_estimates = []
        term_estimates = []
        illegal_estimates = []
        term_states = {x: 0 for x in range(n * m + 1)}
        non_term_states = {x: 0 for x in range(n * m + 1)}
        illegal_states = {x: 0 for x in range(n * m + 1)}
        for run in range(runs):
            a = MonteCarlo(m, n, k)
            a.simulate_n_games(samples)
            non_term_estimate = a.non_term_estimate
            term_estimate = a.term_estimate
            illegal_estimate = a.illegal_estimate
            non_term_estimates.append(non_term_estimate)
            term_estimates.append(term_estimate)
            illegal_estimates.append(illegal_estimate)
            print(f"Run {run} Complete")
            for turn in range(1, m*n+1):
                term_states[turn] += a.term_states[turn]
                non_term_states[turn] += a.non_term_states[turn]
                illegal_states[turn] += a.illegal_states[turn]
        print(non_term_estimates)

        f.write(f"non_term_estimates = {non_term_estimates} \n")
        non_term_mean = sum(non_term_estimates) / len(non_term_estimates)
        nonterm_means.append(non_term_mean)
        # nonterm_sterror = sem(non_term_estimates)
        # nonterm_sterrors.append(nonterm_sterror)
        # f.write(f"\nMean:{non_term_mean} \nStandard Error: {nonterm_sterror}\n\n")

        f.write(f"term_estimates = {term_estimates} \n")
        term_mean = sum(term_estimates) / len(term_estimates)
        term_means.append(term_mean)
        # term_sterror = sem(term_estimates)
        # term_sterrors.append(term_sterror)
        # f.write(f"\nMean:{term_mean} \nStandard Error: {term_sterror}\n\n")

        f.write(f"illegal_estimates = {illegal_estimates} \n")
        illegal_mean = sum(illegal_estimates) / len(illegal_estimates)
        illegal_means.append(illegal_mean)
        # illegal_sterror = sem(illegal_estimates)
        # illegal_sterrors.append(illegal_sterror)
        # f.write(f"\nMean:{illegal_mean} \nStandard Error: {illegal_sterror}\n\n")

        a.term_states = term_states
        a.non_term_states = non_term_states
        a.illegal_states = illegal_states

        states_per_turn = [a.states_per_turn[turn] for turn in range(m*n+1)]

        nonterm_prop = [estimate_nonterm_proportions(a)[turn] for turn in range(m*n+1)]
        term_prop = [estimate_term_proportions(a)[turn] for turn in range(m*n+1)]
        illegal_prop = [estimate_illegal_proportions(a)[turn] for turn in range(m*n+1)]

        nonterm_per_turn = [states_per_turn[turn]*nonterm_prop[turn] for turn in range(m*n + 1)]
        term_per_turn = [states_per_turn[turn]*term_prop[turn] for turn in range(m*n + 1)]
        illegal_per_turn = [states_per_turn[turn]*illegal_prop[turn] for turn in range(m*n + 1)]

        f.write(f'\nstates_per_turn = {states_per_turn}\n')
        f.write(f'\nnon_terminal_per_turn = {nonterm_per_turn}')
        f.write(f'\nterm_per_turn = {term_per_turn}')
        f.write(f'\nillegal_per_turn = {illegal_per_turn}\n')
        f.write(f'\nnonterm_prop = {nonterm_prop}')
        f.write(f'\nterm_prop = {term_prop}')
        f.write(f'\nillegal_prop = {illegal_prop}')
    #
    # with open(f'k<=m<={n}.txt', 'a') as f:
    #
    #     f.write(f'\n\nnonterm_means = {nonterm_means}')
    #     f.write(f'\nnonterm_sterrors = {nonterm_sterrors}')
    #     f.write(f'\n\nterm_means = {term_means}')
    #     f.write(f'\nterm_sterrors = {term_sterrors}')
    #     f.write(f'\n\nillegal_means = {illegal_means}')
    #     f.write(f'\n\nillegal_means = {illegal_sterrors}')
    #
    # #
    # #
    # #
