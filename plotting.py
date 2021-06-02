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


def plot_proportions(nonterm_prop: List[float], term_prop: List[float], illegal_prop: List[float], m: int, n: int,
                     k: int):
    plt.title(label=f"m={m}, n={n}, k={k}")
    plt.plot(nonterm_prop)
    plt.plot(term_prop)
    plt.plot(illegal_prop)
    labels = ["Non Terminal", "Terminal", "Illegal"]
    plt.legend(labels=labels)
    plt.ylabel("Proportion of States")
    plt.xlabel("Turn")
    plt.show()


def plot_log_states(statespt: List[float], non_termpt: List[float], termpt: List[float], illegalpt: List[float], m: int,
                    n: int, k: int):
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


def plot_states(statespt: List[float], non_termpt: List[float], termpt: List[float], illegal_pt: List[float], m: int,
                n: int, k: int):
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


def csv_results_to_dicts(filepath: str) -> List:
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
            first_term =  math.log(state_probability)
            second_term = -b*gamma*math.log(mn/a)
        else:
            first_term = math.log(state_probability)
            second_term = math.log(1-state_probability)
        if state_probability == 0 or state_probability == 1:
            print(state_probability, a,b, gamma, mn)
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
        if state_probability == 0 or state_probability == 1:
            print(state_probability, a,b, gamma, mn)
        res = state_count * first_term + (total-state_count) * second_term
    else:
        print("Invalid LL Predictor")
        raise
    return res / total if total else res


def six_param_nll(vars: numpy.array, states_by_k: List, state_type: str):
    """
    :param vars: optimization parameters
    :param states_by_k:
    :param nonterms_by_k:
    :return:
    """

    alpha = vars[0]
    b = vars[1]
    c = vars[2]
    a_vals = []
    total_nll = 0
    for i in range(len(states_by_k)):
        a_vals.append(power_law(i + 3, alpha, c))
    for i in range(len(states_by_k)):
        total_nll += loglikelihood_sum(a_vals[i], b, states_by_k[i][0], states_by_k[i][1], state_type)
    return total_nll


def eight_param_nll(vars: numpy.array, states_by_k: List, state_type: str):
    """
    :param vars: optimization parameters
    :param states_by_k:
    :param nonterms_by_k:
    :return:
    """

    alpha = vars[0]
    b = vars[1]
    c = vars[2]
    beta = vars[3]
    a_vals = []
    total_nll = 0
    for i in range(len(states_by_k)):
        a_vals.append(power_law_plus_beta(i + 3, alpha, c, beta))
    for i in range(len(states_by_k)):
        total_nll += loglikelihood_sum(a_vals[i], b, states_by_k[i][0], states_by_k[i][1], state_type)
    return total_nll


def four_param_nll(vars: numpy.array, illegal_by_k: List, nonterms_by_k: List):
    """
    :param vars: optimization parameters
    :param illegal_by_k:
    :param nonterms_by_k:
    :return:
    """

    alpha = vars[0]
    b = vars[1]
    c1 = vars[2]
    c2 = vars[3]
    gamma = vars[4]
    illegal_a_vals = []
    nonterm_a_vals = []
    total_illegal_nll = 0
    total_nonterm_nll = 0
    for i in range(len(illegal_by_k)):
        illegal_a_vals.append(power_law(i + 3, alpha, c1))
        nonterm_a_vals.append(power_law(i + 3, alpha, c2))
    for i in range(len(illegal_by_k)):
        total_illegal_nll += loglikelihood_sum(illegal_a_vals[i], b, illegal_by_k[i][0], illegal_by_k[i][1], "illegal")
        total_nonterm_nll += loglikelihood_sum(nonterm_a_vals[i], b, nonterms_by_k[i][0], nonterms_by_k[i][1],
                                               "nonterm")
    return total_nonterm_nll + total_illegal_nll


def five_param_nll(vars: numpy.array, illegal_by_k: List, nonterms_by_k: List):
    """
    :param vars: optimization parameters
    :param illegal_by_k:
    :param nonterms_by_k:
    :return:
    """

    alpha = vars[0]
    b = vars[1]
    c1 = vars[2]
    c2 = vars[3]
    beta = vars[4]
    illegal_a_vals = []
    nonterm_a_vals = []
    total_illegal_nll = 0
    total_nonterm_nll = 0
    for i in range(len(illegal_by_k)):
        illegal_a_vals.append(power_law_plus_beta(i + 3, alpha, c1, beta))
        nonterm_a_vals.append(power_law_plus_beta(i + 3, alpha, c2, beta))
    for i in range(len(illegal_by_k)):
        total_illegal_nll += loglikelihood_sum(illegal_a_vals[i], b, illegal_by_k[i][0], illegal_by_k[i][1], "illegal")
        total_nonterm_nll += loglikelihood_sum(nonterm_a_vals[i], b, nonterms_by_k[i][0], nonterms_by_k[i][1],
                                               "nonterm")
    return total_nonterm_nll + total_illegal_nll


def five_param_fit(x0: numpy.array, illegal_by_k: List, nonterms_by_k: List):
    best = scipy.optimize.minimize(five_param_nll, x0=x0, args=(illegal_by_k, nonterms_by_k), method='Powell',
                                   bounds=((0, None), (0, None), (0, None), (0, None), (0, None)))
    for i in range(1):
        print(i)
        alpha = numpy.random.random_sample() * 10
        b = numpy.random.random_sample() * 10
        c1 = numpy.random.random_sample()
        c2 = numpy.random.random_sample()
        beta = numpy.random.random_sample() * 10
        x0[1] = alpha
        x0[2] = b
        x0[2] = c1
        x0[3] = c2
        x0[4] = beta
        new_res = scipy.optimize.minimize(five_param_nll, x0=x0, args=(illegal_by_k, nonterms_by_k), method='Powell',
                                          bounds=((0, None), (0, None), (0, None), (0, None), (0, None)))
        best = new_res if new_res.fun < best.fun else best
    return best


def four_param_fit(x0: numpy.array, illegal_by_k: List, nonterms_by_k: List):
    best = scipy.optimize.minimize(four_param_nll, x0=x0, args=(illegal_by_k, nonterms_by_k), method='Powell',
                                   bounds=((0, None), (0, None), (0, None), (0, None)))
    print(best)
    for i in range(20):
        print(i)
        alpha = numpy.random.random_sample() * 10
        b = numpy.random.random_sample() * 10
        c1 = numpy.random.random_sample() * 2
        c2 = numpy.random.random_sample() * 2
        gamma = numpy.random.random_sample() * 10
        x0[1] = alpha
        x0[2] = b
        x0[2] = c1
        x0[3] = c2
        x0[4] = gamma
        new_res = scipy.optimize.minimize(four_param_nll, x0=x0, args=(illegal_by_k, nonterms_by_k), method='Powell',
                                          bounds=((0, None), (0, None), (0, None), (0, None)))
        best = new_res if new_res.fun < best.fun else best
        print(best)
    return best

def six_param_fit(x0: numpy.array, states_by_k: List, state_type: str):
    best = scipy.optimize.minimize(six_param_nll, x0=x0, args=(states_by_k, state_type), method='Powell',
                                   bounds=((0, None), (0, None), (0, None)))
    for i in range(1):
        print(i)
        alpha = numpy.random.random_sample() * 10
        b = numpy.random.random_sample() * 10
        c = numpy.random.random_sample()
        x0[0] = alpha
        x0[1] = b
        x0[2] = c
        new_res = scipy.optimize.minimize(six_param_nll, x0=x0, args=(states_by_k, state_type), method='Powell',
                                          bounds=((0, None), (0, None), (0, None)))
        best = new_res if new_res.fun < best.fun else best
    return best

def eight_param_fit(x0: numpy.array, states_by_k: List, state_type: str):
    best = scipy.optimize.minimize(eight_param_nll, x0=x0, args=(states_by_k, state_type), method='Powell',
                                   bounds=((0, None), (0, None), (0, None), (0,None)))
    for i in range(20):
        print(i)
        alpha = numpy.random.random_sample() * 10
        b = numpy.random.random_sample() * 10
        c = numpy.random.random_sample()
        beta = numpy.random.random_sample() * 5
        x0[0] = alpha
        x0[1] = b
        x0[2] = c
        x0[3] = beta
        new_res = scipy.optimize.minimize(eight_param_nll, x0=x0, args=(states_by_k, state_type), method='Powell',
                                          bounds=((0, None), (0, None), (0, None), (0, None)))
        best = new_res if new_res.fun < best.fun else best
    return best



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
    a0_arr = numpy.random.uniform(1, 200, 2)
    b0_arr = numpy.random.uniform(1, 3, 2)
    # x0 = numpy.array([a0, b0])
    gamma0_arr = numpy.random.uniform(0, 2, 2)
    x0 = numpy.array([a0, b0, gamma0])
    # x0 = numpy.array([a0])
    # x0_arr = numpy.random.uniform(0, 600, 10)

    best = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor), method= 'L-BFGS-B', bounds=((1,None),(0.1,None),(0.1, None)))
    # best = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor), method='Powell', bounds=((0,None),(0,None)))

    for x0 in itertools.product(a0_arr, b0_arr, gamma0_arr):
        print(x0)
    # for x0 in itertools.product(a0_arr, b0_arr):
    #     print(best)
    # for x0 in x0_arr:
        x0 = numpy.array([x0])
        res = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor), method= 'L-BFGS-B', bounds=((1,None),(0.1,None),(0.1,None)))
        # res = scipy.optimize.minimize(loglikelihood_sum, x0=x0, args=(mn_list, state_list, predictor),
        #                               method='Powell')
    #
        best = res if (res.fun < best.fun and not numpy.isnan(res.fun)) or numpy.isnan(best.fun) and res.success else best
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
        estimated_prop = [likelihood_function(a, b, gamma, x) for x in prop_by_k[i][0]]
        prediction_coords = zip(prop_by_k[i][0], estimated_prop)
        prediction_coords = sorted(prediction_coords)
        prediction_coords = zip(*prediction_coords)
        prediction_coords = [list(z) for z in prediction_coords]
        mle_preds.append(prediction_coords[1])
        if do_plot:
            plt.scatter(prop_by_k[i][0], prop_by_k[i][1], label=f'{i + 3}')
            plt.plot(prediction_coords[0], prediction_coords[1])
        with open(f'{state_type}coefficients.csv', 'a') as resultfile:
            rowwriter = csv.writer(resultfile, delimiter=',')
            # row = [i + 3, a, b]
            row = [i + 3, a, b, gamma]
            print(row)
            row = [str(entry) for entry in row]
            print(row)
            rowwriter.writerow(row)
    return mle_preds


def arrange_record_plot_terminal(do_plot: bool, terms_by_k: List, term_prop_by_k: List, mle_illegal_preds: List,
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


if __name__ == "__main__":
    # headers:
    # m,n,k,mn,nonterm_mean,nonterm_sterror,term_mean,term_sterror,illegal_mean,illegal_sterror
    results = csv_results_to_dicts("resulttable.csv")
    nonterms_by_k = []
    terms_by_k = []
    illegal_by_k = []
    total_states = []

    colors = ['#ffffe5', '#f7fcb9', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32']
    colors.reverse()

    for k in range(3, 11):
        nonterms_by_k.append(variables_by_k(k, "mn", "nonterm_mean", results))
        terms_by_k.append(variables_by_k(k, "mn", "term_mean", results))
        illegal_by_k.append(variables_by_k(k, "mn", "illegal_mean", results))
    nonterm_prop_by_k = mn_by_prop(nonterms_by_k)
    term_prop_by_k = mn_by_prop(terms_by_k)
    illegal_prop_by_k = mn_by_prop(illegal_by_k)

    """Four parameter fitting"""
    #
    # x0 = [3.37, 3.15, .43, .36]
    # x0 = numpy.array(x0)
    # res = four_param_fit(x0, illegal_by_k, nonterms_by_k)
    # illegal_fig = plt.figure(1)
    # plt.title(label="Illegal proportion 4 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    # for i in range(len(illegal_prop_by_k)):
    #     plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i + 3}')
    #     illegal_a = power_law(i+3, res.x[0], res.x[2])
    #     illegal_b = res.x[1]
    #     line = numpy.linspace(4, 350, 100)
    #     illegal_preds = [illegal_prop_likelihood(illegal_a, illegal_b, x) for x in line]
    #     plt.plot(line, illegal_preds)
    # print(res)
    # nonterm_fig = plt.figure(2)
    # plt.title(label="Non-Terminal proportion 4 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    # for i in range(len(nonterm_prop_by_k)):
    #     plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i + 3}')
    #     nonterm_a = power_law(i+3, res.x[0], res.x[3])
    #     nonterm_b = res.x[1]
    #     line = numpy.linspace(4, 350, 100)
    #     nonterm_preds = [nonterm_prop_likelihood(nonterm_a, nonterm_b, x) for x in line]
    #     plt.plot(line, nonterm_preds)

    """Five parameter fitting"""
    # x0 = [3.37, 3.15, .43, .36, 5]
    # x0 = numpy.array(x0)
    # res = five_param_fit(x0, illegal_by_k, nonterms_by_k)
    # illegal_fig = plt.figure(1)
    # plt.title(label="Illegal proportion 5 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    # for i in range(len(illegal_prop_by_k)):
    #     plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i + 3}')
    #     illegal_a = power_law_plus_beta(i + 3, res.x[0], res.x[2], res.x[4])
    #     illegal_b = res.x[1]
    #     line = numpy.linspace(4, 350, 100)
    #     illegal_preds = [illegal_prop_likelihood(illegal_a, illegal_b, x) for x in line]
    #     plt.plot(line, illegal_preds)
    # print(res)
    # nonterm_fig = plt.figure(2)
    # plt.title(label="Non-Terminal proportion 5 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    # for i in range(len(nonterm_prop_by_k)):
    #     plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i + 3}')
    #     nonterm_a = power_law_plus_beta(i+3, res.x[0], res.x[3], res.x[4])
    #     nonterm_b = res.x[1]
    #     line = numpy.linspace(4, 350, 100)
    #     nonterm_preds = [nonterm_prop_likelihood(nonterm_a, nonterm_b, x) for x in line]
    #     plt.plot(line, nonterm_preds)

    """Six parameter fitting"""

    # x0 = [3.37, 3.15, .40]
    # x0 = numpy.array(x0)
    # illegal_res = six_param_fit(x0, illegal_by_k, "illegal")
    # illegal_fig = plt.figure(1)
    # plt.title(label="Illegal proportion 6 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    # for i in range(len(illegal_prop_by_k)):
    #     plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i + 3}')
    #     illegal_a = power_law(i + 3, illegal_res.x[0], illegal_res.x[2])
    #     illegal_b = illegal_res.x[1]
    #     line = numpy.linspace(4, 140, 100)
    #     illegal_preds = [illegal_prop_likelihood(illegal_a, illegal_b, x) for x in line]
    #     plt.plot(line, illegal_preds)
    # print(illegal_res)
    # nonterm_res = six_param_fit(x0, nonterms_by_k, "nonterm")
    # nonterm_fig = plt.figure(2)
    # plt.title(label="Non-Terminal proportion 6 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    # for i in range(len(nonterm_prop_by_k)):
    #     plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i + 3}')
    #     nonterm_a = power_law(i+3, nonterm_res.x[0], nonterm_res.x[2])
    #     nonterm_b = nonterm_res.x[1]
    #     line = numpy.linspace(4, 140, 100)
    #     nonterm_preds = [nonterm_prop_likelihood(nonterm_a, nonterm_b, x) for x in line]
    #     plt.plot(line, nonterm_preds)
    # print(nonterm_res)

    """Eight parameter fitting"""
    #
    # x0 = [3.37, 3.15, .40, 1.0]
    # x0 = numpy.array(x0)
    # illegal_res = eight_param_fit(x0, illegal_by_k, "illegal")
    # illegal_fig = plt.figure(1)
    # for i in range(len(illegal_prop_by_k)):
    #     plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i + 3}')
    #     illegal_a = power_law_plus_beta(i + 3, illegal_res.x[0], illegal_res.x[2], illegal_res.x[3])
    #     illegal_b = illegal_res.x[1]
    #     line = numpy.linspace(4, 340, 100)
    #     illegal_preds = [illegal_prop_likelihood(illegal_a, illegal_b, x) for x in line]
    #     plt.plot(line, illegal_preds)
    # print(illegal_res)
    # plt.title(label="Illegal proportion 8 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    # nonterm_res = eight_param_fit(x0, nonterms_by_k, "nonterm")
    # nonterm_fig = plt.figure(2)
    # for i in range(len(nonterm_prop_by_k)):
    #     plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i + 3}')
    #     nonterm_a = power_law_plus_beta(i + 3, nonterm_res.x[0], nonterm_res.x[2], nonterm_res.x[3])
    #     nonterm_b = nonterm_res.x[1]
    #     line = numpy.linspace(4, 340, 100)
    #     nonterm_preds = [nonterm_prop_likelihood(nonterm_a, nonterm_b, x) for x in line]
    #     plt.plot(line, nonterm_preds)
    # print(nonterm_res)
    # plt.title(label="Non-Terminal proportion 8 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()

    """Four parameter plotting with Terminal"""
    # orig_parmas = [3.32975196, 3.20643512, 0.46437459, 0.34516418]
    # orig_parmas = [3.34104257, 3.1691086, 0.45716397, 0.33992056]
    # nonterm_preds_by_i = []
    # illegal_preds_by_i = []
    # for i in range(len(nonterm_prop_by_k)):
    #     # plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i + 3}')
    #     nonterm_a = power_law(i+3, orig_parmas[0], orig_parmas[3])
    #     nonterm_b = orig_parmas[1]
    #     line = numpy.linspace(4, 140, 100)
    #     nonterm_preds = [nonterm_prop_likelihood(nonterm_a, nonterm_b, x) for x in line]
    #     nonterm_preds_by_i.append(nonterm_preds)
    #     # plt.plot(line, nonterm_preds)
    #
    # for i in range(len(illegal_prop_by_k)):
    #     # plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i + 3}')
    #     illegal_a = power_law(i+3, orig_parmas[0], orig_parmas[2])
    #     illegal_b = orig_parmas[1]
    #     line = numpy.linspace(4, 140, 100)
    #     illegal_preds = [illegal_prop_likelihood(illegal_a, illegal_b, x) for x in line]
    #     illegal_preds_by_i.append(illegal_preds)
    #     # plt.plot(line, illegal_preds)
    #
    # for i in range(len(illegal_prop_by_k)):
    #     plt.scatter(term_prop_by_k[i][0], term_prop_by_k[i][1], label=f'{i + 3}')
    #     line = numpy.linspace(4, 140, 100)
    #     term_preds = [1-illegal_preds_by_i[i][x]-nonterm_preds_by_i[i][x] for x in range(100)]
    #     plt.plot(line, term_preds)

    "Five parameter plotting wtih Terminal"

    # orig_parmas = [3.64419626, 2.98890855, 0.29977529, 0.22194919, 0.89511715]
    # orig_parmas = [3.68043533, 2.99884897, 0.28240566, 0.20683945, 1.43161333]
    # nonterm_preds_by_i = []
    # illegal_preds_by_i = []
    # for i in range(len(nonterm_prop_by_k)):
    #     # plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i + 3}')
    #     nonterm_a = power_law_plus_beta(i+3, orig_parmas[0], orig_parmas[3], orig_parmas[4])
    #     nonterm_b = orig_parmas[1]
    #     line = numpy.linspace(4, 140, 100)
    #     nonterm_preds = [nonterm_prop_likelihood(nonterm_a, nonterm_b, x) for x in line]
    #     nonterm_preds_by_i.append(nonterm_preds)
    #     # plt.plot(line, nonterm_preds)
    #
    # for i in range(len(illegal_prop_by_k)):
    #     # plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i + 3}')
    #     illegal_a = power_law_plus_beta(i+3, orig_parmas[0], orig_parmas[2], orig_parmas[4])
    #     illegal_b = orig_parmas[1]
    #     line = numpy.linspace(4, 140, 100)
    #     illegal_preds = [illegal_prop_likelihood(illegal_a, illegal_b, x) for x in line]
    #     illegal_preds_by_i.append(illegal_preds)
    #     # plt.plot(line, illegal_preds)
    #
    # for i in range(len(illegal_prop_by_k)):
    #     plt.scatter(term_prop_by_k[i][0], term_prop_by_k[i][1], label=f'{i + 3}')
    #     line = numpy.linspace(4, 140, 100)
    #     term_preds = [1-illegal_preds_by_i[i][x]-nonterm_preds_by_i[i][x] for x in range(100)]
    #     plt.plot(line, term_preds)
    #
    """Eight parameter plotting wtih Terminal"""
    #
    # orig_illegal_paramas = [3.99776214, 2.86283706, 0.18438861, 1.81723774]
    # orig_nonterm_paramas = [4.35239377, 2.54641668, 0.07404511, 3.1253808 ]
    # nonterm_preds_by_i = []
    # illegal_preds_by_i = []
    # nonterm_fig = plt.figure(1)
    # for i in range(len(nonterm_prop_by_k)):
    #     plt.scatter(nonterm_prop_by_k[i][0], nonterm_prop_by_k[i][1], label=f'{i + 3}')
    #     nonterm_a = power_law_plus_beta(i+3, orig_nonterm_paramas[0], orig_nonterm_paramas[2], orig_nonterm_paramas[3])
    #     nonterm_b = orig_nonterm_paramas[1]
    #     line = numpy.linspace(4, 340, 100)
    #     nonterm_preds = [nonterm_prop_likelihood(nonterm_a, nonterm_b, x) for x in line]
    #     nonterm_preds_by_i.append(nonterm_preds)
    #     plt.plot(line, nonterm_preds)
    # plt.title(label="Non-Terminal proportion 8 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    #
    # illegal_fig = plt.figure(2)
    # for i in range(len(illegal_prop_by_k)):
    #     plt.scatter(illegal_prop_by_k[i][0], illegal_prop_by_k[i][1], label=f'{i + 3}')
    #     illegal_a = power_law_plus_beta(i+3, orig_illegal_paramas[0], orig_illegal_paramas[2], orig_illegal_paramas[3])
    #     illegal_b = orig_illegal_paramas[1]
    #     line = numpy.linspace(4, 340, 100)
    #     illegal_preds = [illegal_prop_likelihood(illegal_a, illegal_b, x) for x in line]
    #     illegal_preds_by_i.append(illegal_preds)
    #     plt.plot(line, illegal_preds)
    # plt.title(label="Illegal proportion 8 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()
    #
    # terminal_fig = plt.figure(3)
    # for i in range(len(illegal_prop_by_k)):
    #     plt.scatter(term_prop_by_k[i][0], term_prop_by_k[i][1], label=f'{i + 3}')
    #     line = numpy.linspace(4, 340, 100)
    #     term_preds = [1-illegal_preds_by_i[i][x]-nonterm_preds_by_i[i][x] for x in range(100)]
    #     plt.plot(line, term_preds)
    # plt.title(label="Terminal proportion 8 parameters")
    # plt.ylabel("Proportion")
    # plt.xlabel("m*n")
    # plt.yscale("linear")
    # plt.legend()

    """Old powerlaw fitting sequence"""

    #
    # MLE_nested_all_k(nonterm_prop_by_k, nonterms_by_k, "nonterm")z
    # MLE_nested_all_k(illegal_prop_by_k, illegal_by_k, "illegal")

    mle_illegal_preds = arrange_record_and_plot_mle("illegal_burr", True, illegal_prop_by_k, illegal_by_k, illegal_burr_likelihood)
    # mle_nonterm_preds = arrange_record_and_plot_mle("nonterm_burr", True, nonterm_prop_by_k, nonterms_by_k, nonterm_burr_likelihood)
    # mle_term_preds = arrange_record_plot_terminal(False, terms_by_k, term_prop_by_k, mle_illegal_preds, mle_nonterm_preds)
    # # #
    # illegal_a_and_k_values = return_and_plot_a_coefficients("illegal", False)
    # nonterm_a_and_k_values = return_and_plot_a_coefficients("nonterm", True)

    # illegal_powerlaw_fit = fit_coefficient_power_law(illegal_a_and_k_values[0], illegal_a_and_k_values[1])[0]
    # nonterm_powerlaw_fit = fit_coefficient_power_law(nonterm_a_and_k_values[0], nonterm_a_and_k_values[1])[0]
    #
    # illegal_powerlaw_preds = [power_law(x, illegal_powerlaw_fit[0], illegal_powerlaw_fit[1]) for x in numpy.linspace(3, 10, 20)]
    # line = numpy.linspace(3, 10, 20)
    # nonterm_powerlaw_preds = [power_law(x, nonterm_powerlaw_fit[0], nonterm_powerlaw_fit[1]) for x in numpy.linspace(3, 10, 20)]
    #
    # # plt.plot(line, illegal_powerlaw_preds)
    # plt.plot(line, nonterm_powerlaw_preds)


    # plt.title(label="Nonterm proportion Burr dist max parameters")
    plt.title(label="Illegal proportion Burr dist max parameters")
    plt.ylabel("Proportion")
    plt.xlabel("m*n")
    plt.yscale("linear")
    plt.legend()
    #





    plt.show()
