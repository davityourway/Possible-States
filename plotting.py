import csv
import pandas
import math
import scipy
import numpy
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm

from scipy.interpolate import interp1d
from scipy import optimize
from typing import List, Tuple, Dict, Callable
from math import factorial
from montecarlostates import find_states_per_turn


def plot_proportions(nonterm_prop: List[float], term_prop: List[float], illegal_prop: List[float], m: int, n: int,
                     k: int):
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.title(label=f"m={m}, n={n}, k={k}")
    colors = ['#1b9e77', '#d95f02', '#7570b3']
    x = [x for x in range(len(nonterm_prop))]
    plt.scatter(x, nonterm_prop,color = colors[0], s=30)
    plt.scatter(x, term_prop,color=colors[1], s=30)
    plt.scatter(x, illegal_prop,color = colors[2], s=30)
    labels = ["Non Terminal", "Terminal", "Illegal"]
    legend = plt.legend(labels=labels,bbox_to_anchor=(0,.5), loc="center left", title="State Types")
    for handle in legend.legendHandles:
        handle._sizes = [30]
    plt.ylabel("Proportion of States")
    plt.xlabel("Turn")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.rcParams['axes.labelsize'] = 100
    plt.rcParams['axes.titlesize'] = 100
    plt.show()


def plot_log_states(statespt: List[float], non_termpt: List[float], termpt: List[float], illegalpt: List[float], m: int,
                    n: int, k: int):
    colors = ["#e7298a", '#1b9e77', '#d95f02', '#7570b3']
    plt.figure(figsize=(3,5))
    plt.rcParams.update({'font.size': 14})
    plt.title(label=f"m={m}, n={n}, k={k}")
    plt.plot(statespt, linestyle='dashed')
    plt.plot(non_termpt, color=colors[0])
    plt.plot(termpt, color=colors[1])
    plt.plot(illegalpt, color=colors[2])
    plt.yscale("log")
    labels = ["Total States", "Non-Terminal", "Terminal", "Illegal"]
    plt.legend(labels=labels, title="State Types")
    plt.ylabel("Log(States)")
    plt.xlabel("Turn")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()


def plot_states(statespt: List[float], non_termpt: List[float], termpt: List[float], illegal_pt: List[float], m: int,
                n: int, k: int):
    # plt.title(label=f"m={m}, n={n}, k={k}")
    plt.rcParams.update({'font.size': 12})
    x = [x for x in range(len(statespt))]
    colors = ["#e7298a", '#1b9e77', '#d95f02', '#7570b3']
    plt.scatter(x, statespt, color=colors[0], s=2.5)
    plt.scatter(x, non_termpt, color=colors[1], s=2.5)
    plt.scatter(x, termpt, color=colors[2], s=2.5)
    plt.scatter(x, illegal_pt, color=colors[3], s=2.5)
    # plt.plot(non_termpt)
    # plt.plot(termpt)
    # plt.plot(illegal_pt)
    labels = ["Possible States", "Non-Terminal", "Terminal", "Illegal"]
    legend = plt.legend(labels=labels, title="State Types")
    for handle in legend.legendHandles:
        handle._sizes = [20]
    plt.ylabel("States")
    plt.xlabel("Turn")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.major.formatter._useMathText = True
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.rcParams.update({'font.size': 12})
    plt.show()


def csv_results_to_dicts(filepath: str) -> List:
    with open(filepath, newline='') as csvfile:
        res_reader = csv.DictReader(csvfile)
        results = []
        for row in res_reader:
            results.append(row)
        return results

def csv_results_to_tupledict(filepath:str) -> dict:
    with open(filepath, newline='') as csvfile:
        result_dict = {}
        res_reader = csv.DictReader(csvfile)
        for row in res_reader:
            result_dict[(row["m"], row["n"], row["k"])] = row
        return result_dict


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
    for turn in range(1, positions + 1):
        m = turn // 2
        n = turn - m
        total += factorial(positions) // (factorial(n) * factorial(m) * factorial(positions - turn))
    return total


def mn_by_states(max_mn: int):
    mn = []
    states = []
    for i in range(9, max_mn + 1):
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
            prop_series.append(state_series[j] / mn_states)
        k_prop_list.append((mn_series, prop_series))
    return k_prop_list


def illegal_prop_likelihood(a: float, b: float, x: float):
    denom = 1 + (x / a) ** -b
    return 1 / denom


def nonterm_prop_likelihood(a: float, b: float, x: float):
    denom = 1 + (x / a) ** -b
    return 1 - 1 / denom

def illegal_burr_likelihood(a: float, b: float, gamma:float, x: float):
    likelihood = 1 - (1 + (x / a) ** b)**-gamma
    return likelihood

def nonterm_burr_likelihood(a: float, b: float, gamma:float, x: float):
    likelihood = (1 + (x / a) ** b)**-gamma
    # print(likelihood, x, a, b, gamma)
    return likelihood


# def loglikelihood_at_mn(mn: float, state_count: float, a: float, b: float, predictor: str):
def loglikelihood_at_mn(mn: float, state_count: float, a: float, b: float, gamma:float, predictor: str):
    # "log likelihood LL(parameters) = sum_i [ nNT_i * log f(x_i) +  (ntot_i - nNT_i) * log(1-f(x_i)) ] + constant"
    total = find_states_mn(mn)
    if predictor == "illegal":
        # res = state_count * -math.log(1 + (mn / a) ** -b) + (total - state_count) * -math.log(1 + (mn / a) ** b)
        res = state_count * -math.log(1 + ((mn - gamma) / a) ** -b) + (total - state_count) * -math.log(1 + ((mn-gamma) / a) ** b)
    elif predictor == "nonterm":
        # res = state_count * -math.log(1 + (mn / a) ** b) + (total - state_count) * -math.log(1 + (mn / a) ** -b)
        res = state_count * -math.log(1 + ((mn-gamma) / a) ** b) + (total - state_count) * -math.log(1 + ((mn-gamma) / a) ** -b)
    elif predictor == "illegal_burr":
        state_probability = illegal_burr_likelihood(a, b, gamma, mn)
        if state_probability == 0:
            first_term = math.log(gamma) + b*math.log(mn/a)
            second_term = math.log(1 - state_probability)
        elif state_probability == 1:
            first_term = math.log(state_probability)
            second_term = -b*gamma*math.log(mn/a)
        else:
            first_term = math.log(state_probability)
            second_term = math.log(1-state_probability)
        res = state_count * first_term + (total-state_count) * second_term

    elif predictor == "nonterm_burr":
        state_probability = nonterm_burr_likelihood(a, b, gamma, mn)
        if state_probability == 0:
            first_term = -b * gamma * math.log(mn / a)
            second_term = math.log(1 - state_probability)
        elif state_probability == 1:
            first_term = math.log(state_probability)
            second_term = math.log(gamma) + b * math.log(mn / a)
        else:
            first_term = math.log(state_probability)
            second_term = math.log(1 - state_probability)
        res = state_count * first_term + (total-state_count) * second_term
    else:
        print("Invalid LL Predictor")
        raise
    return res / total if total else res

# def loglikelihood_sum(a, b, mn_list: List, state_list: List, predictor: str):
def loglikelihood_sum(vars, mn_list: List, state_list: List, predictor: str):
    a = vars[0]
    b = vars[1]
    gamma = vars[2]
    total = 0
    # predictor = illegal_prop_likelihood if predictor == "illegal" else nonterm_prop_likelihood
    for i in range(len(mn_list)):
        ll = loglikelihood_at_mn(mn_list[i], state_list[i], a, b, gamma, predictor)
        # ll = loglikelihood_at_mn(mn_list[i], state_list[i], a, b, predictor)
        total += ll
    return -total / len(mn_list)

def minimize_wrapper(param_tuple: Tuple):
    x0 = param_tuple[0]
    mn_list = param_tuple[1]
    state_list = param_tuple[2]
    predictor = param_tuple[3]
    res = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor), method='L-BFGS-B',
                            bounds=((0, None), (1, None), (0, None)))
    return res



def MLE_given_k(mn_list: List, state_list: List, a0: float, b0: float, gamma0: float, predictor: str):
    a0_arr = numpy.random.uniform(1, 500, 3)
    b0_arr = numpy.random.uniform(1, 3, 3)
    gamma0_arr = numpy.random.uniform(0, 2, 3)
    x0 = numpy.array([a0, b0, gamma0])
    best = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor), method= 'L-BFGS-B', bounds=((1,None),(0.001,None),(0.001, None)))
    for x0 in itertools.product(a0_arr, b0_arr, gamma0_arr):
        print(x0)
        x0 = numpy.array([x0])
        res = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor), method= 'L-BFGS-B', bounds=((1,None),(0.001,None),(0.001,None)))
        best = res if (res.fun < best.fun and not numpy.isnan(res.fun)) or numpy.isnan(best.fun) and res.success or not best.success else best
    print(best)
    return best


def MLE_given_b_k(mn_list: List, state_list: List, a0: float, b0: float, predictor: str):
    x0 = numpy.array([a0])
    x0_arr = numpy.random.uniform(0, 600, 10)
    best = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(b0, mn_list, state_list, predictor),
                                   method='L-BFGS-B', bounds=((0, None),))
    # best = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(b0, mn_list, state_list, predictor), method= 'Powell', bounds=((0,None),))
    for x0 in x0_arr:
        x0 = numpy.array([x0])
        res = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(b0, mn_list, state_list, predictor),
                                      method='L-BFGS-B', bounds=((0, None),))
        # res = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(b0, mn_list, state_list, predictor), method= 'Powell', bounds=((0,None),))
        best = res if res.fun < best.fun else best
    return best


"""implement mle for all k"""


def MLE_all_k_given_b(b: float, prop_by_k: List, states_by_k: List, predictor: str):
    total = 0
    for i in range(len(states_by_k)):
        mn_vals = states_by_k[i][0]
        state_vals = states_by_k[i][1]
        optimizer_res = MLE_given_b_k(mn_vals, state_vals, 20, b, predictor)
        print(optimizer_res.x)
        print(optimizer_res.fun)
        total += optimizer_res.fun / len(mn_vals)
    print(total, b)
    return total


def MLE_nested_all_k(prop_by_k: List, states_by_k: List, predictor: str):
    x0 = numpy.array([3.15])
    res = scipy.optimize.minimize(MLE_all_k_given_b, x0=x0, args=(prop_by_k, states_by_k, predictor), method='Powell',
                                  bounds=((0, None),))
    print(res.x)
    return res.fun


def arrange_record_and_plot_mle(state_type: str, do_plot: bool, prop_by_k: List, states_by_k: List,
                                likelihood_function: Callable):
    """

    :param state_type:
    :param do_plot:
    :param prop_by_k:
    :param states_by_k:
    :param likelihood_function:
    :return:
    plots MLE with each a and b as optimized parameters
    """
    mle_preds = []
    params_by_k = []
    with open(f'{state_type}coefficients.csv', 'a') as resultfile:
        headerwriter = csv.writer(resultfile, delimiter=',')
        header = ["k", "a", "b", "gamma"]
        # header = ["k", "a", "b"]
        headerwriter.writerow(header)
    for i in range(len(prop_by_k)):
        mn_vals = states_by_k[i][0]
        state_vals = states_by_k[i][1]
        optimizer_res = MLE_given_k(mn_vals, state_vals, 20, 2, 2, state_type)
        # optimizer_res = MLE_given_k(mn_vals, state_vals, 20, 6, state_type)
        print(optimizer_res)
        print(optimizer_res.fun)
        a = optimizer_res.x[0]
        b = optimizer_res.x[1]
        gamma = optimizer_res.x[2]
        # estimated_prop = [likelihood_function(a, b, x) for x in prop_by_k[i][0]]
        mn_array = numpy.linspace(0, 380, 500)
        estimated_prop = [likelihood_function(a, b, gamma, x) for x in mn_array]
        # prediction_coords = zip(prop_by_k[i][0], estimated_prop)
        # prediction_coords = sorted(prediction_coords)
        # prediction_coords = zip(*prediction_coords)
        # prediction_coords = [list(z) for z in prediction_coords]
        mle_preds.append(estimated_prop)
        params_by_k.append((a,b,gamma))
        if do_plot:
            plt.scatter(prop_by_k[i][0], prop_by_k[i][1], label=f'{i + 2}')
            # plt.plot(prediction_coords[0], prediction_coords[1])
            plt.plot(mn_array, estimated_prop)
        with open(f'{state_type}coefficients.csv', 'a') as resultfile:
            rowwriter = csv.writer(resultfile, delimiter=',')
            # row = [i + 3, a, b]
            row = [i + 2, a, b, gamma]
            print(row)
            row = [str(entry) for entry in row]
            print(row)
            rowwriter.writerow(row)
    return mle_preds, params_by_k


def arrange_record_plot_terminal_by_points(do_plot: bool, terms_by_k: List, term_prop_by_k: List, mle_illegal_preds: List,
                                           mle_nonterm_preds: List):
    mle_term_preds = []
    for i in range(len(terms_by_k)):
        pred_coordinates = zip(term_prop_by_k[i][0], term_prop_by_k[i][1])
        pred_coordinates = sorted(pred_coordinates)
        pred_coordinates = zip(*pred_coordinates)
        pred_coordinates = [list(z) for z in pred_coordinates]
        nontermatk = mle_nonterm_preds[i]
        illegalatk = mle_illegal_preds[i]
        mle_term_pred_k = [1 - nontermatk[i] - illegalatk[i] for i in range(len(nontermatk))]
        mle_term_preds.append(mle_term_pred_k)
        if do_plot:
            plt.plot(pred_coordinates[0], mle_term_pred_k)
            plt.scatter(pred_coordinates[0], pred_coordinates[1], label=f'{i + 3}')
    return mle_term_preds

def plot_terminal_by_parameter(terms_by_k: List, illegal_params: List, nonterm_params: List):
    mn_array = numpy.linspace(0, 380, 500)
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'lines.markersize': 4})
    for i in range(len(illegal_params)):
        nonterm_k_para = nonterm_params[i]
        illegal_k_para = illegal_params[i]
        term_preds = [1 - nonterm_burr_likelihood(float(nonterm_k_para['a']), float(nonterm_k_para['b']), float(nonterm_k_para['gamma']), x) -
                      illegal_burr_likelihood(float(illegal_k_para['a']), float(illegal_k_para['b']), float(illegal_k_para['gamma']), x) for x in mn_array]
        plt.plot(mn_array, term_preds)
        plt.scatter(terms_by_k[i][0], terms_by_k[i][1], label=f'{i+2}')


def plot_state_by_parameter(state_count_by_k: List, params: List, likelihood_function: Callable):
    mn_array = numpy.linspace(0, 380, 500)
    # colors = ['#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4']
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'lines.markersize': 4})
    for i in range(len(params)):
        i_param = {key:float(params[i][key]) for key in params[i].keys()}
        estimated_prop = [likelihood_function(i_param["a"], i_param["b"], i_param["gamma"], x) for x in mn_array]
        plt.plot(mn_array, estimated_prop)
        plt.scatter(state_count_by_k[i][0], state_count_by_k[i][1], label=f'{i+2}')


def return_and_plot_a_coefficients(state_type: str, do_plot: bool):
    results = csv_results_to_dicts(f"{state_type}coefficients.csv")
    a_values = []
    k_values = []
    b = results[0]["b"]
    for result in results:
        k_values.append(float(result["k"]))
        a_values.append(float(result["a"]))
    if do_plot:
        plt.scatter(k_values, a_values)
    return a_values, k_values


def power_law(k: float, alpha: float, c: float):
    return c * k ** alpha


def power_law_plus_beta(k: float, alpha: float, c: float, beta: float):
    return c * k ** alpha + beta


def fit_coefficient_power_law(a_values: List, k_values: List):
    res = scipy.optimize.curve_fit(power_law, k_values, a_values)
    # res = scipy.optimize.least_squares(power_law)
    print(res)
    return res


def maxturns_arrays():
    zero_maxes = []
    zero_maxturns = []
    one_maxes = []
    one_maxturns = []
    two_maxes = []
    two_maxturns = []
    for i in range(2, 200):
        turns = [x / i for x in range(1, i + 1)]
        states = find_states_per_turn(i)
        states = [states[x] for x in range(1, i + 1)]
        if i % 3 == 0:
            zero_maxes.append(max(states))
            zero_maxturns.append((states.index(zero_maxes[-1]) + 1) / (len(turns)))
        elif i % 3 == 1:
            one_maxes.append(max(states))
            one_maxturns.append((states.index(one_maxes[-1]) + 1) / (len(turns)))
        elif i % 3 == 2:
            two_maxes.append(max(states))
            two_maxturns.append((states.index(two_maxes[-1]) + 1) / (len(turns)))

    return zero_maxes, zero_maxturns, one_maxes, one_maxturns, two_maxes, two_maxturns



if __name__ == "__main__":

    # headers:
    # m,n,k,mn,nonterm_mean,nonterm_sterror,term_mean,term_sterror,illegal_mean,illegal_sterror
    results = csv_results_to_dicts("resulttable.csv")
    nonterms_by_k = []
    terms_by_k = []
    illegal_by_k = []
    total_states = []

    colors = ['#1b9e77','#d95f02','#7570b3']
    colors.reverse()

    for k in range(2, 11):
        nonterms_by_k.append(variables_by_k(k, "mn", "nonterm_mean", results))
        terms_by_k.append(variables_by_k(k, "mn", "term_mean", results))
        illegal_by_k.append(variables_by_k(k, "mn", "illegal_mean", results))
    nonterm_prop_by_k = mn_by_prop(nonterms_by_k)
    term_prop_by_k = mn_by_prop(terms_by_k)
    illegal_prop_by_k = mn_by_prop(illegal_by_k)

    """Old powerlaw fitting sequence"""

    #
    # MLE_nested_all_k(nonterm_prop_by_k, nonterms_by_k, "nonterm")
    # MLE_nested_all_k(illegal_prop_by_k, illegal_by_k, "illegal")

    # mle_illegal_preds, illegal_params = arrange_record_and_plot_mle("illegal_burr", True, illegal_prop_by_k, illegal_by_k, illegal_burr_likelihood)
    # mle_nonterm_preds, nonterm_params = arrange_record_and_plot_mle("nonterm_burr", True, nonterm_prop_by_k, nonterms_by_k, nonterm_burr_likelihood)
    # mle_term_preds = arrange_record_plot_terminal_by_points(True, terms_by_k, term_prop_by_k, mle_illegal_preds, mle_nonterm_preds)

    """Plots from recorded fitting"""
    #
    # nonterm_params = csv_results_to_dicts('nonterm_burrcoefficients.csv')
    # illegal_params = csv_results_to_dicts('illegal_burrcoefficients.csv')
    # plot_terminal_by_parameter(term_prop_by_k, illegal_params, nonterm_params)

    """fitting sequence"""

    # plot_state_by_parameter(illegal_prop_by_k, illegal_params, illegal_burr_likelihood)
    # plot_state_by_parameter(nonterm_prop_by_k, nonterm_params,nonterm_burr_likelihood)
    # # #
    # illegal_a_and_k_values = return_and_plot_a_coefficients("illegal", False)
    # nonterm_a_and_k_values = return_and_plot_a_coefficients("nonterm", True)
    #
    # illegal_powerlaw_fit = fit_coefficient_power_law(illegal_a_and_k_values[0], illegal_a_and_k_values[1])[0]
    # nonterm_powerlaw_fit = fit_coefficient_power_law(nonterm_a_and_k_values[0], nonterm_a_and_k_values[1])[0]
    #
    # illegal_powerlaw_preds = [power_law(x, illegal_powerlaw_fit[0], illegal_powerlaw_fit[1]) for x in numpy.linspace(3, 10, 20)]
    # line = numpy.linspace(3, 10, 20)
    # nonterm_powerlaw_preds = [power_law(x, nonterm_powerlaw_fit[0], nonterm_powerlaw_fit[1]) for x in numpy.linspace(3, 10, 20)]

    # plt.plot(line, illegal_powerlaw_preds)
    # plt.plot(line, nonterm_powerlaw_preds)

    """Maxturns plot"""
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'lines.markersize': 4})

    zero_maxes, zero_maxturns, one_maxes, one_maxturns, two_maxes, two_maxturns = maxturns_arrays()
    plt.scatter(zero_maxturns, zero_maxes, color=colors[0], s=5, label = "mn = 0 mod 3")
    plt.scatter(one_maxturns, one_maxes, color=colors[1], s=5, label = "mn = 1 mod 3")
    plt.scatter(two_maxturns, two_maxes, color=colors[2], s=5, label = "mn = 2 mod 3")
    plt.plot(zero_maxturns, zero_maxes, color=colors[0])
    plt.plot(one_maxturns, one_maxes, color=colors[1])
    plt.plot(two_maxturns, two_maxes, color=colors[2])
    # plt.plot(maxturns, maxes)


    # plt.title(label="Maximum State Count", size=14)
    # plt.title(label="Possible States per turn")
    plt.ylabel("Maximum states")
    plt.xlabel("Occupancy at maximum")
    # plt.ylabel("Proportion ")
    # plt.xlabel("Board Size")
    # plt.title("Terminal Proportion by k", size=14)
    plt.yscale("log")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # # scientific notation
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.yaxis.set_major_formatter(formatter)


    # plt.yscale("log")
    legend = plt.legend(bbox_to_anchor=(.5,1), loc="upper left", title="Turn")
    for handle in legend.legendHandles:
        handle._sizes = [30]

    #

    plt.rcParams["figure.figsize"] = (5, 3)
    plt.show()
