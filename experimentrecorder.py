import _io
import csv
import multiprocessing as mp
import argparse
from typing import Tuple
import functools
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
    non_term_estimates_by_turn = []
    term_estimates_by_turn = []
    illegal_estimates_by_turn = []
    non_term_raw_list = []
    term_raw_list = []
    illegal_raw_list = []

    for run in range(runs):
        # Run monte carlo simulation
        a = MonteCarlo(m, n, k)
        a.simulate_n_games(samples)
        # record run estimates
        non_term_estimate = a.non_term_estimate
        term_estimate = a.term_estimate
        illegal_estimate = a.illegal_estimate
        non_term_estimates.append(non_term_estimate)
        term_estimates.append(term_estimate)
        illegal_estimates.append(illegal_estimate)
        # Record Raw counts
        non_term_raw_list.append(a.non_term_states)
        term_raw_list.append(a.term_states)
        illegal_raw_list.append(a.illegal_states)
        print(f"M={m}, N={n}, K={k}: Run {run+1}")

        states_per_turn = [a.states_per_turn[turn] for turn in range(m * n + 1)]

        nonterm_prop = [estimate_nonterm_proportions(a)[turn] for turn in range(m * n + 1)]
        term_prop = [estimate_term_proportions(a)[turn] for turn in range(m * n + 1)]
        illegal_prop = [estimate_illegal_proportions(a)[turn] for turn in range(m * n + 1)]

        nonterm_per_turn = [states_per_turn[turn] * nonterm_prop[turn] for turn in range(m * n + 1)]
        term_per_turn = [states_per_turn[turn] * term_prop[turn] for turn in range(m * n + 1)]
        illegal_per_turn = [states_per_turn[turn] * illegal_prop[turn] for turn in range(m * n + 1)]

        non_term_estimates_by_turn.append(nonterm_per_turn)
        term_estimates_by_turn.append(term_per_turn)
        illegal_estimates_by_turn.append(illegal_per_turn)

    # Calculate total estimates
    non_term_mean = sum(non_term_estimates) / len(non_term_estimates)
    nonterm_sterror = sem(non_term_estimates)
    term_mean = sum(term_estimates) / len(term_estimates)
    term_sterror = sem(term_estimates)
    illegal_mean = sum(illegal_estimates) / len(illegal_estimates)
    illegal_sterror = sem(illegal_estimates)
    # assemble raw totals by turn
    non_term_total = functools.reduce(lambda c, d: [c[i] + d[i] for i in range(len(c))], non_term_raw_list)
    term_total = functools.reduce(lambda c, d: [c[i] + d[i] for i in range(len(c))], term_raw_list)
    illegal_total = functools.reduce(lambda c, d: [c[i] + d[i] for i in range(len(c))], illegal_raw_list)
    # assemble estimates by turn
    nonterm_turn_estimates_mean_byturn = []
    term_turn_estimates_mean_byturn = []
    illegal_turn_estimates_mean_byturn = []
    nonterm_turn_estimates_sterror_byturn = []
    term_turn_estimates_sterror_byturn = []
    illegal_turn_estimates_sterror_byturn = []
    for turn in range(len(non_term_estimates_by_turn[0])):
        nonterm_turn_estimates = []
        term_turn_estimates = []
        illegal_turn_estimates = []
        for run in range(len(non_term_estimates_by_turn)):
            nonterm_turn_estimates.append(non_term_estimates_by_turn[run][turn])
            term_turn_estimates.append(term_estimates_by_turn[run][turn])
            illegal_turn_estimates.append(illegal_estimates_by_turn[run][turn])

        nonterm_turn_mean = sum(nonterm_turn_estimates) / len(nonterm_turn_estimates)
        nonterm_turn_sterror = sem(nonterm_turn_estimates)
        nonterm_turn_estimates_mean_byturn.append(nonterm_turn_mean)
        nonterm_turn_estimates_sterror_byturn.append(nonterm_turn_sterror)

        term_turn_mean = sum(term_turn_estimates) / len(term_turn_estimates)
        term_turn_sterror = sem(term_turn_estimates)
        term_turn_estimates_mean_byturn.append(term_turn_mean)
        term_turn_estimates_sterror_byturn.append(term_turn_sterror)

        illegal_turn_mean = sum(illegal_turn_estimates) / len(illegal_turn_estimates)
        illegal_turn_sterror = sem(illegal_turn_estimates)
        illegal_turn_estimates_mean_byturn.append(illegal_turn_mean)
        illegal_turn_estimates_sterror_byturn.append(illegal_turn_sterror)

    result_row = [m, n, k, m*n, non_term_mean, nonterm_sterror, term_mean, term_sterror, illegal_mean, illegal_sterror,
                  non_term_total, term_total, illegal_total, nonterm_turn_estimates_mean_byturn,
                  nonterm_turn_estimates_sterror_byturn, term_turn_estimates_mean_byturn,
                  term_turn_estimates_sterror_byturn, illegal_turn_estimates_mean_byturn,
                  illegal_turn_estimates_sterror_byturn]
    with open(fname, 'a') as resultfile:
        resultwriter = csv.writer(resultfile, delimiter=',')
        resultwriter.writerow(result_row)
    pass

def mp_worker(fvalues: Tuple):
    run_and_record_to_csv(fvalues[3], fvalues[0], fvalues[1], fvalues[2], 10, 100000)


if __name__ == "__main__":
    fpath = 'resulttablev2.csv'
    # with open(fpath, 'a') as resultfile:
    #     headerwriter = csv.writer(resultfile, delimiter=',')
    #     header = ["m", "n", "k", "mn", "nonterm_mean", "nonterm_sterror", "term_mean", "term_sterror", "illegal_mean",
    #               "illegal_sterror", "non_term_total_by_turn", "term_total_by_turn", "illegal_total_by_turn",
    #               "non_term_est_by_turn", "non_term_sterror_by_turn", "term_est_by_turn", "term_sterror_by_turn",
    #               "illegal_est_by_turn", "illegal_sterror_by_turn"]
    #     headerwriter.writerow(header)
    # result_boards = [(x, y, z, fpath) for x in range(15, 18) for y in range(15, 18) for z in range(2, 18) if (x <= y) and
    #                  (z <= x)]

    # REMEMBR RUN 2->11 for K values
    result_boards = [(10, 19, 10), (12, 19, 12), (10, 11, 6), (9, 15, 9), (13, 18, 6), (14, 16, 7), (5, 15, 4), (7, 15, 6), (10, 14, 7), (8, 17, 2), (3, 16, 2), (15, 19, 4), (17, 19, 6), (10, 15, 8), (15, 19, 13), (12, 15, 10), (10, 18, 4), (17, 19, 15), (9, 14, 3), (3, 17, 3), (6, 13, 2), (11, 19, 2), (11, 19, 11), (10, 19, 5), (9, 11, 6), (12, 19, 7), (9, 15, 4), (5, 18, 5), (10, 11, 10), (14, 16, 2), (14, 16, 11), (6, 16, 3), (14, 19, 13), (16, 19, 15), (7, 18, 3), (10, 14, 2), (15, 18, 7), (9, 18, 5), (6, 17, 4), (5, 13, 2), (8, 17, 6), (11, 15, 9), (10, 15, 3), (15, 19, 8), (12, 15, 5), (17, 19, 10), (5, 14, 3), (10, 18, 8), (7, 14, 5), (9, 14, 7), (6, 13, 6), (11, 19, 6), (13, 17, 10), (12, 19, 2), (12, 18, 6), (5, 17, 4), (7, 17, 6), (14, 16, 6), (4, 13, 3), (8, 13, 4), (9, 17, 8), (14, 19, 8), (16, 19, 10), (15, 18, 2), (7, 18, 7), (10, 14, 6), (15, 18, 11), (9, 18, 9), (11, 15, 4), (15, 19, 3), (17, 19, 5), (10, 15, 7), (15, 19, 12), (10, 18, 3), (12, 15, 9), (17, 19, 14), (9, 14, 2), (11, 19, 10), (6, 12, 5), (13, 17, 5), (8, 12, 7), (8, 16, 5), (12, 18, 10), (9, 17, 3), (10, 13, 9), (6, 16, 2), (8, 13, 8), (4, 16, 3), (14, 19, 3), (16, 19, 5), (14, 19, 12), (16, 19, 14), (7, 18, 2), (15, 18, 6), (9, 18, 4), (10, 14, 10), (2, 17, 2), (15, 18, 15), (4, 17, 4), (11, 15, 8), (13, 16, 8), (10, 17, 2), (17, 18, 13), (5, 16, 2), (11, 19, 5), (7, 16, 4), (8, 12, 2), (13, 17, 9), (14, 15, 10), (3, 11, 2), (14, 18, 6), (12, 18, 5), (5, 17, 3), (7, 17, 5), (10, 13, 4), (4, 13, 2), (8, 13, 3), (9, 17, 7), (6, 16, 6), (3, 12, 3), (14, 19, 7), (16, 19, 9), (10, 14, 5), (15, 18, 10), (5, 13, 5), (11, 15, 3), (13, 16, 3), (3, 19, 2), (11, 18, 5), (13, 16, 12), (17, 18, 8), (10, 17, 6), (17, 18, 17), (7, 13, 3), (12, 17, 8), (9, 13, 5), (13, 17, 4), (6, 12, 4), (8, 12, 6), (13, 17, 13), (14, 15, 5), (8, 16, 4), (14, 15, 14), (14, 18, 10), (12, 18, 9), (9, 17, 2), (10, 13, 8), (4, 16, 2), (14, 19, 2), (16, 19, 4), (14, 19, 11), (6, 19, 3), (16, 19, 13), (8, 19, 5), (15, 18, 5), (5, 12, 4), (10, 16, 9), (7, 12, 6), (12, 16, 11), (4, 19, 3), (9, 12, 8), (13, 16, 7), (17, 18, 3), (11, 18, 9), (17, 18, 12), (12, 17, 3), (10, 17, 10), (7, 13, 7), (12, 17, 12), (7, 16, 3), (9, 13, 9), (9, 16, 5), (14, 15, 9), (14, 18, 5), (12, 18, 4), (14, 18, 14), (5, 17, 2), (7, 17, 4), (10, 13, 3), (9, 17, 6), (16, 19, 8), (6, 11, 3), (16, 18, 12), (8, 11, 5), (10, 16, 4), (12, 16, 6), (9, 12, 3), (3, 15, 3), (13, 16, 2), (11, 18, 4), (13, 16, 11), (17, 18, 7), (10, 17, 5), (17, 18, 16), (7, 13, 2), (12, 17, 7), (9, 13, 4), (5, 16, 5), (2, 12, 2), (7, 16, 7), (4, 12, 4), (9, 16, 9), (14, 15, 4), (13, 19, 3), (6, 15, 5), (14, 18, 9), (8, 15, 7), (13, 19, 12), (8, 18, 3), (5, 11, 2), (6, 19, 2), (7, 11, 4), (8, 19, 4), (16, 18, 7), (16, 18, 16), (11, 17, 7), (5, 12, 3), (10, 16, 8), (7, 12, 5), (12, 16, 10), (4, 19, 2), (9, 12, 7), (13, 16, 6), (17, 18, 2), (11, 18, 8), (17, 18, 11), (13, 15, 10), (12, 17, 2), (7, 16, 2), (9, 16, 4), (4, 11, 3), (14, 18, 4), (8, 15, 2), (9, 19, 6), (13, 19, 7), (6, 18, 5), (14, 17, 8), (3, 14, 2), (8, 18, 7), (10, 12, 6), (11, 16, 10), (7, 19, 3), (16, 18, 2), (6, 19, 6), (6, 11, 2), (8, 19, 8), (16, 18, 11), (8, 11, 4), (11, 17, 2), (11, 17, 11), (10, 16, 3), (12, 16, 5), (9, 12, 2), (11, 18, 3), (13, 15, 5), (6, 14, 3), (8, 14, 5), (13, 18, 10), (13, 19, 2), (6, 15, 4), (4, 14, 3), (8, 15, 6), (13, 19, 11), (14, 17, 3), (8, 18, 2), (14, 17, 12), (11, 16, 5), (5, 19, 5), (10, 12, 10), (2, 15, 2), (7, 19, 7), (4, 15, 4), (7, 11, 3), (16, 18, 6), (16, 18, 15), (11, 17, 6), (5, 12, 2), (10, 16, 7), (7, 12, 4), (12, 16, 9), (9, 12, 6), (10, 19, 9), (10, 11, 5), (12, 19, 11), (9, 15, 8), (13, 15, 9), (13, 18, 5), (5, 15, 3), (7, 15, 5), (4, 11, 2), (9, 19, 5), (13, 19, 6), (6, 18, 4), (14, 17, 7), (8, 18, 6), (10, 12, 5), (11, 16, 9), (7, 19, 2), (5, 11, 5), (2, 18, 2), (7, 11, 7), (16, 18, 10), (11, 17, 10), (3, 17, 2), (10, 19, 4), (9, 11, 5), (12, 19, 6), (9, 15, 3), (13, 15, 4), (5, 18, 4), (10, 11, 9), (6, 14, 2), (13, 15, 13), (8, 14, 4), (13, 18, 9), (14, 16, 10), (4, 14, 2), (9, 19, 9), (14, 17, 2), (14, 17, 11), (6, 17, 3), (11, 16, 4), (8, 17, 5), (7, 11, 2), (10, 15, 2), (15, 19, 7), (12, 15, 4), (17, 19, 9), (5, 14, 2), (10, 18, 7), (7, 14, 4), (9, 14, 6), (6, 13, 5), (10, 19, 8), (4, 18, 4), (9, 11, 9), (10, 11, 4), (12, 19, 10), (9, 15, 7), (13, 15, 8), (13, 18, 4), (6, 14, 6), (8, 14, 8), (13, 18, 13), (14, 16, 5), (14, 16, 14), (5, 15, 2), (7, 15, 4), (9, 19, 4), (7, 18, 6), (14, 17, 6), (9, 18, 8), (11, 16, 8), (15, 19, 2), (17, 19, 4), (10, 15, 6), (15, 19, 11), (10, 18, 2), (12, 15, 8), (17, 19, 13), (11, 19, 9), (10, 19, 3), (9, 11, 4), (12, 19, 5), (9, 15, 2), (13, 15, 3), (5, 18, 3), (10, 11, 8), (8, 14, 3), (13, 18, 8), (14, 16, 9), (8, 13, 7), (9, 18, 3), (10, 14, 9), (15, 18, 14), (6, 17, 2), (4, 17, 3), (8, 17, 4), (11, 16, 3), (11, 15, 7), (15, 19, 6), (12, 15, 3), (17, 19, 8), (10, 15, 10), (15, 19, 15), (10, 18, 6), (12, 15, 12), (7, 14, 3), (17, 19, 17), (9, 14, 5), (6, 13, 4), (11, 19, 4), (13, 17, 8), (4, 18, 3), (9, 11, 8), (10, 11, 3), (8, 16, 8), (14, 16, 4), (8, 13, 2), (14, 16, 13), (6, 16, 5), (3, 12, 2), (14, 19, 6), (7, 18, 5), (10, 14, 4), (15, 18, 9), (9, 18, 7), (6, 17, 6), (5, 13, 4), (3, 13, 3), (8, 17, 8), (11, 15, 2), (11, 15, 11), (17, 19, 3), (10, 15, 5), (15, 19, 10), (12, 15, 7), (17, 19, 12), (5, 14, 5), (10, 18, 10), (7, 14, 7), (9, 14, 9), (11, 19, 8), (6, 12, 3), (13, 17, 3), (8, 12, 5), (13, 17, 12), (8, 16, 3), (14, 15, 13), (12, 18, 8), (10, 13, 7), (8, 13, 6), (16, 19, 3), (14, 19, 10), (16, 19, 12), (15, 18, 4), (9, 18, 2), (10, 14, 8), (15, 18, 13), (4, 17, 2), (8, 17, 3), (3, 16, 3), (11, 15, 6), (15, 19, 5), (12, 15, 2), (17, 19, 7), (10, 18, 5), (7, 14, 2), (9, 14, 4), (10, 17, 9), (7, 13, 6), (12, 17, 11), (11, 19, 3), (9, 13, 8), (13, 17, 7), (14, 15, 8), (8, 16, 7), (12, 18, 3), (14, 18, 13), (12, 18, 12), (7, 17, 3), (10, 13, 2), (9, 17, 5), (6, 16, 4), (14, 19, 5), (16, 19, 7), (14, 19, 14), (16, 19, 16), (7, 18, 4), (10, 14, 3), (15, 18, 8), (9, 18, 6), (5, 13, 3), (11, 15, 10), (17, 19, 2), (3, 15, 2), (13, 16, 10), (17, 18, 6), (10, 17, 4), (17, 18, 15), (12, 17, 6), (9, 13, 3), (5, 16, 4), (13, 17, 2), (6, 12, 2), (7, 16, 6), (4, 12, 3), (8, 12, 4), (9, 16, 8), (13, 17, 11), (8, 16, 2), (14, 15, 3), (14, 15, 12), (14, 18, 8), (12, 18, 7), (5, 17, 5), (2, 13, 2), (7, 17, 7), (10, 13, 6), (4, 13, 4), (8, 13, 5), (9, 17, 9), (16, 19, 2), (14, 19, 9), (16, 19, 11), (8, 19, 3), (15, 18, 3), (6, 11, 6), (15, 18, 12), (3, 18, 3), (8, 11, 8), (11, 15, 5), (13, 16, 5), (11, 18, 7), (17, 18, 10), (10, 17, 8), (7, 13, 5), (12, 17, 10), (9, 13, 7), (9, 16, 3), (13, 17, 6), (6, 12, 6), (8, 12, 8), (14, 15, 7), (8, 16, 6), (14, 18, 3), (12, 18, 2), (14, 18, 12), (12, 18, 11), (7, 17, 2), (9, 17, 4), (10, 13, 10), (2, 16, 2), (4, 16, 4), (14, 19, 4), (16, 19, 6), (6, 19, 5), (8, 19, 7), (8, 11, 3), (10, 16, 2), (12, 16, 4)]
    result_boards = [(board[0], board[1], board[2], fpath) for board in result_boards]
    # run_and_record_to_csv(fpath, 3, 3, 3, 10, 100000)
    # for board in result_boards:
    #     run_and_record_to_csv(fpath, board[0], board[1], board[2], 10, 100000)
    with mp.Pool(8) as p:
        p.map(mp_worker, result_boards)
