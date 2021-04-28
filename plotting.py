import csv
import pandas
import math
import scipy
import numpy
import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.interpolate import interp1d
from scipy import optimize
from typing import List, Tuple, Dict, Callable
from math import factorial

def plot_proportions(nonterm_prop: List[float], term_prop: List[float], illegal_prop: List[float], m:int, n:int, k:int):
    plt.title(label=f"m={m}, n={n}, k={k}")
    plt.plot(nonterm_prop)
    plt.plot(term_prop)
    plt.plot(illegal_prop)
    labels = ["Non Terminal", "Terminal", "Illegal"]
    plt.legend(labels=labels)
    plt.ylabel("Proportion of States")
    plt.xlabel("Turn")
    plt.show()


def plot_log_states(statespt: List[float], non_termpt: List[float], termpt: List[float], illegalpt: List[float], m:int, n:int, k:int):
    plt.title(label=f"m={m}, n={n}, k={k}")
    plt.plot(statespt, linestyle='dashed')
    plt.plot(non_termpt)
    plt.plot(termpt)
    plt.plot(illegalpt)
    plt.yscale("log")
    labels = ["Possible States", "Non Terminal", "Terminal", "Illegal"]
    plt.legend(labels=labels)
    plt.ylabel("Log(States)")
    plt.xlabel("Turn")
    plt.show()


def plot_states(statespt: List[float], non_termpt: List[float], termpt: List[float], illegal_pt: List[float], m:int, n:int, k:int):
    plt.title(label=f"m={m}, n={n}, k={k}")
    plt.plot(statespt)
    plt.plot(non_termpt)
    plt.plot(termpt)
    plt.plot(illegal_pt)
    labels = ["Possible States", "Non Terminal", "Terminal", "Illegal"]
    plt.legend(labels=labels)
    plt.ylabel("States")
    plt.xlabel("Turn")
    plt.show()

def csv_results_to_dicts(filepath:str) -> List:
    with open(filepath, newline='') as csvfile:
        res_reader = csv.DictReader(csvfile)
        results = []
        for row in res_reader:
            results.append(row)
        return results

def variables_by_k(k: int, variable1: str, variable2: str, results: List[Dict]) -> Tuple[List, List]:
    var1 = []
    var2 = []
    for result in results:
        if result["k"] == str(k):
            var1.append(float(result[variable1]))
            var2.append(float(result[variable2]))
    return var1, var2

def find_states_mn(positions: int):
    """
    determines how many possible states (of all kinds, even illegal) can come from a given mn
    using the forumla from https://psyarxiv.com/rhq5j pg.19 of Supplemental Material

    :param positions: total number of positions that can be filled
    :return: total number of states
    """
    total = 1
    positions = int(positions)
    for turn in range(1, positions+1):
        m = turn//2
        n = turn - m
        total += factorial(positions) // (factorial(n) * factorial(m) * factorial(positions - turn))
    return total

def mn_by_states(max_mn: int):
    mn = []
    states = []
    for i in range(9, max_mn+1):
        mn.append(i)
        states.append(find_states_mn(i))
    return mn, states

def mn_by_prop(k_list: List[Tuple[List, List]]):
    k_prop_list = []
    for i in range(len(k_list)):
        prop_series = []
        mn_series = k_list[i][0]
        state_series = k_list[i][1]
        for j in range(len(k_list[i][0])):
            mn_states = find_states_mn(int(mn_series[j]))
            prop_series.append(state_series[j]/mn_states)
        k_prop_list.append((mn_series, prop_series))
    return k_prop_list


def illegal_prop_likelihood(a: float, b: float, x: float):
    denom = 1 + (x/a) ** -b
    return 1 / denom


def nonterm_prop_likelihood(a: float, b: float, x: float):
    denom = 1 + (x/a) ** -b
    return 1 - 1 / denom


def loglikelihood_at_mn(mn: float, state_count: float, a: float, b:float, predictor: str):
    # "log likelihood LL(parameters) = sum_i [ nNT_i * log f(x_i) +  (ntot_i - nNT_i) * log(1-f(x_i)) ] + constant"
    total = find_states_mn(mn)
    #
    # predval = predictor(a, b, mn)
    if predictor == "illegal":
        res = state_count * -math.log(1 + (mn/a) ** -b) + (total - state_count)*-math.log(1 + (mn/a) ** b)
    elif predictor == "nonterm":
        res = state_count * -math.log(1 + (mn/a) ** b) + (total - state_count)*-math.log(1 + (mn/a) ** -b)
    else:
        print("Invalid LL Predictor")
        raise
    return res/total if total else res


def loglikelihood_sum(vars, mn_list: List, state_list: List, predictor: str):
    total = 0
    # predictor = illegal_prop_likelihood if predictor == "illegal" else nonterm_prop_likelihood
    for i in range(len(mn_list)):
        ll = loglikelihood_at_mn(mn_list[i], state_list[i], vars[0], vars[1], predictor)
        total += ll
    return -total


def MLE_given_k(mn_list: List, state_list: List, a0: float, b0: float, predictor: str):
    a0_arr = numpy.random.uniform(0, 300, 10)
    b0_arr = numpy.random.uniform(0, 10, 10)
    x0 = numpy.array([a0, b0])
    best = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor), method= 'L-BFGS-B', bounds= ((0, None), (0, None)))
    for x0 in itertools.product(a0_arr, b0_arr):
        x0 = numpy.array(x0)
        res = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor), method= 'L-BFGS-B', bounds= ((0, None), (0, None)))
        best = res if res.fun < best.fun else best
    print(best)
    return best







if __name__ == "__main__":
    # headers:
    # m,n,k,mn,nonterm_mean,nonterm_sterror,term_mean,term_sterror,illegal_mean,illegal_sterror
    results = csv_results_to_dicts("resulttable.csv")
    nonterms_by_k = []
    terms_by_k = []
    illegal_by_k = []
    total_states = []
    mle_nonterm_preds = []
    mle_illegal_preds = []

    colors = ['#ffffe5','#f7fcb9','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32']
    colors.reverse()

    for k in range(3,11):
        nonterms_by_k.append(variables_by_k(k, "mn", "nonterm_mean", results))
        terms_by_k.append(variables_by_k(k, "mn", "term_mean", results))
        illegal_by_k.append(variables_by_k(k, "mn", "illegal_mean", results))
    nonterm_prop_by_k = mn_by_prop(nonterms_by_k)
    term_prop_by_k = mn_by_prop(terms_by_k)
    illegal_prop_by_k = mn_by_prop(illegal_by_k)

    with open('illegalcoefficients.csv', 'a') as resultfile:
        headerwriter = csv.writer(resultfile, delimiter=',')
        header = ["k", "functionvalue", "a", "b"]
        headerwriter.writerow(header)

    for i in range(8):
        xs = illegal_by_k[i][0]
        ys = illegal_by_k[i][1]
        opt = MLE_given_k(xs, ys, 20, 6, "illegal")
        print(opt)
        print(opt.fun)
        a = opt.x[0]
        b = opt.x[1]
        y = [illegal_prop_likelihood(a, b, x) for x in illegal_prop_by_k[i][0]]
        coordinates = zip(illegal_by_k[i][0], y)
        coordinates = sorted(coordinates)
        xy = zip(*coordinates)
        xy = [list(z) for z in xy]
        mle_illegal_preds.append(xy[1])
        plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i+3}')
        plt.plot(xy[0], xy[1])
        with open('illegalcoefficients.csv', 'a') as resultfile:
            rowwriter = csv.writer(resultfile, delimiter=',')
            row = [i+3, opt.fun, opt.x[0], opt.x[1]]
            row = [str(entry) for entry in row]
            rowwriter.writerow(row)
    # for i in range(8):
    #     xs = nonterms_by_k[i][0]
    #     ys = nonterms_by_k[i][1]
    #     opt = MLE_given_k(xs, ys, 20, 6, "nonterm")
    #     a = opt.x[0]
    #     b = opt.x[1]
    #     print(opt)
    #     y = [nonterm_prop_likelihood(a, b, x) for x in nonterm_prop_by_k[i][0]]
    #     coordinates = zip(nonterms_by_k[i][0], y)
    #     coordinates = sorted(coordinates)
    #     xy = zip(*coordinates)
    #     xy = [list(z) for z in xy]
    #     mle_nonterm_preds.append(xy[1])
    #     plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i+3}')
    #     plt.plot(xy[0], xy[1])
    #     with open('nontermcoefficients.csv', 'a') as resultfile:
    #         rowwriter = csv.writer(resultfile, delimiter=',')
    #         row = [i+3, opt.fun, opt.x[0], opt.x[1]]
    #         row = [str(entry) for entry in row]
    #         rowwriter.writerow(row)
    # for i in range(8):
    #     coordinates = zip(term_prop_by_k[i][0], term_prop_by_k[i][1])
    #     coordinates = sorted(coordinates)
    #     xy = zip(*coordinates)
    #     xy = [list(z) for z in xy]
    #     nontermatk = mle_nonterm_preds[i]
    #     illegalatk = mle_illegal_preds[i]
    #     mle_term_preds = [1 - nontermatk[i] - illegalatk[i] for i in range(len(nontermatk))]
    #     plt.plot(xy[0], mle_term_preds)
    #     plt.scatter(xy[0], xy[1], label=f'{i + 3}')


    plt.title(label="Illegal States Proportion by k")
    plt.ylabel("Illegal Proportion")
    plt.xlabel("m*n")
    # plt.yscale("log")
    plt.legend()

    plt.show()
