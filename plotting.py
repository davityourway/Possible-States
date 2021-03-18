from typing import List, Tuple, Dict
import csv
import pandas
import math
import matplotlib.pyplot as plt

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
    plt.plot(illegal_per_turn)
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




if __name__ == "__main__":
    # headers:
    # m,n,k,mn,nonterm_mean,nonterm_sterror,term_mean,term_sterror,illegal_mean,illegal_sterror

    results = csv_results_to_dicts("resulttable.csv")
    nonterms_by_k = []
    terms_by_k = []
    illegal_by_k = []
    total_states = []

    colors = ['#ffffe5','#f7fcb9','#d9f0a3','#addd8e','#78c679','#41ab5d','#238443','#005a32']
    colors.reverse()

    for k in range(3,10):
        nonterms_by_k.append(variables_by_k(k, "mn", "nonterm_mean", results))
        terms_by_k.append(variables_by_k(k, "mn", "term_mean", results))
        illegal_by_k.append(variables_by_k(k, "mn", "illegal_mean", results))
    nonterm_prop_by_k = mn_by_prop(nonterms_by_k)
    term_prop_by_k = mn_by_prop(terms_by_k)
    illegal_prop_by_k = mn_by_prop(illegal_by_k)

    # for i in range(len(nonterms_by_k)):
    #     plt.plot(sorted(nonterms_by_k[i][0]), sorted(nonterms_by_k[i][1]), label=f'{i+3}')
    # for i in range(len(terms_by_k)):
    #     plt.plot(sorted(terms_by_k[i][0]), sorted(terms_by_k[i][1]), label=f'{i+3}')
    # for i in range(len(terms_by_k)):
    #     plt.plot(sorted(terms_by_k[i][0]), sorted(terms_by_k[i][1]), label=f'{i+3}')
    # for i in range(len(nonterm_prop_by_k)):
    #     plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i+3}')
    # for i in range(len(term_prop_by_k)):
    #     plt.scatter(term_prop_by_k[i][0], term_prop_by_k[i][1], label=f'{i + 3}')
    for i in range(len(illegal_prop_by_k)):
        plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i + 3}')
    #
    #
    # mn, states = mn_by_states(81)
    # plt.plot(mn, states, label="Max", color = "blue")

    plt.title(label="Illegal States Proportion by k")
    plt.ylabel("Illegal Proportion")
    plt.xlabel("m*n")
    # plt.yscale("log")
    plt.legend()

    plt.show()
