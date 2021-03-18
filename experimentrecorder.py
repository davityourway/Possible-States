import _io
import montecarlostates
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