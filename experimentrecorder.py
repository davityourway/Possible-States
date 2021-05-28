import _io
import csv
import multiprocessing as mp
from typing import Tuple

from scipy.stats import sem
from montecarlostates import MonteCarlo, estimate_term_proportions, estimate_illegal_proportions, estimate_nonterm_proportions


def record_experiment(f: '_io.TextIOWrapper', m: int, n: int, k: int, runs: int, samples: int):
    if runs == 0 or samples == 0 or m == 0 or n == 0 or k == 0:
        print("Invalid Input")
        raise
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
    nonterm_sterror = sem(non_term_estimates)
    f.write(f"\nMean:{non_term_mean} \nStandard Error: {nonterm_sterror}\n\n")

    f.write(f"term_estimates = {term_estimates} \n")
    term_mean = sum(term_estimates) / len(term_estimates)
    term_sterror = sem(term_estimates)
    f.write(f"\nMean:{term_mean} \nStandard Error: {term_sterror}\n\n")

    f.write(f"illegal_estimates = {illegal_estimates} \n")
    illegal_mean = sum(illegal_estimates) / len(illegal_estimates)
    illegal_sterror = sem(illegal_estimates)
    f.write(f"\nMean:{illegal_mean} \nStandard Error: {illegal_sterror}\n\n")

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

def run_and_record_to_csv(fname: str, m: int, n: int, k: int, runs: int, samples: int):
    non_term_estimates = []
    term_estimates = []
    illegal_estimates = []
    for run in range(runs):
        a = MonteCarlo(m, n, k)
        a.simulate_n_games(samples)
        non_term_estimate = a.non_term_estimate
        term_estimate = a.term_estimate
        illegal_estimate = a.illegal_estimate
        non_term_estimates.append(non_term_estimate)
        term_estimates.append(term_estimate)
        illegal_estimates.append(illegal_estimate)
        print(f"M={m}, N={n}, K={k}: Run {run+1}")

    non_term_mean = sum(non_term_estimates) / len(non_term_estimates)
    nonterm_sterror = sem(non_term_estimates)
    term_mean = sum(term_estimates) / len(term_estimates)
    term_sterror = sem(term_estimates)
    illegal_mean = sum(illegal_estimates) / len(illegal_estimates)
    illegal_sterror = sem(illegal_estimates)
    result_row = [m, n, k, m*n, non_term_mean, nonterm_sterror, term_mean, term_sterror, illegal_mean, illegal_sterror]
    with open(fname, 'a') as resultfile:
        resultwriter = csv.writer(resultfile, delimiter=',')
        resultwriter.writerow(result_row)
    pass

def mp_worker(fvalues: Tuple):
    run_and_record_to_csv(fvalues[3], fvalues[0], fvalues[1], fvalues[2], 10, 100000)


if __name__ == "__main__":
    fpath = 'resulttable.csv'
    # with open(fpath, 'a') as resultfile:
    #     headerwriter = csv.writer(resultfile, delimiter=',')
    #     header = ["m", "n", "k", "mn", "nonterm_mean", "nonterm_sterror", "term_mean", "term_sterror", "illegal_mean", "illegal_sterror"]
    #     headerwriter.writerow(header)
    four_boards = [(x, y, 4, fpath) for x in range(10, 20) for y in range(11, 20) if x <= y]
    three_boards = [(x, y, 3, fpath) for x in range(10, 20) for y in range(11, 20) if x <= y]
    five_boards = [(x, y, 5, fpath) for x in range(10, 20) for y in range(16, 20) if x <= y]
    six_boards = [(x, y, 6, fpath) for x in range(10, 20) for y in range(16, 20) if x <= y]
    seven_boards = [(x, y, 7, fpath) for x in range(10, 20) for y in range(11, 20) if x <= y]
    eight_boards = [(x, y, 8, fpath) for x in range(8, 20) for y in range(11, 20) if x <= y]
    nine_boards = [(x, y, 9, fpath) for x in range(9, 20) for y in range(11, 20) if x <= y]
    ten_boards = [(x, y, 10, fpath) for x in range(10, 20) for y in range(11, 20) if x <= y]
    result_boards = eight_boards + nine_boards + ten_boards
    run_and_record_to_csv(fpath, 3, 3, 3, 10, 100000)
    # for board in result_boards:
    #     run_and_record_to_csv(fpath, board[0], board[1], board[2], 10, 100000)
    # with mp.Pool(10) as p:
    #     p.map(mp_worker, result_boards)
