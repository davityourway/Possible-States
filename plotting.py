from typing import List

import matplotlib.pyplot as plt


nonterm_prop = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999256, 0.99851, 0.99547, 0.9925, 0.985228, 0.977854, 0.963502, 0.949406, 0.925282, 0.90204, 0.865616, 0.830848, 0.781722, 0.73565, 0.674938, 0.618348, 0.548386, 0.485982, 0.41343, 0.352048, 0.284456, 0.230248, 0.174414, 0.132942, 0.094208, 0.066804, 0.04327, 0.028518, 0.017046, 0.01032]
term_prop = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000744, 0.000746, 0.003784, 0.003716, 0.01105, 0.011062, 0.02531, 0.025018, 0.049038, 0.047572, 0.08398, 0.08032, 0.128972, 0.121024, 0.179664, 0.165566, 0.229496, 0.20473, 0.265984, 0.226852, 0.277222, 0.223808, 0.256004, 0.193114, 0.204412, 0.14364, 0.1394, 0.089406, 0.078828, 0.04589]
illegal_prop = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000744, 0.000746, 0.003784, 0.003722, 0.011084, 0.011188, 0.025576, 0.02568, 0.050388, 0.050404, 0.088832, 0.089306, 0.143326, 0.145398, 0.216086, 0.222118, 0.309288, 0.320586, 0.4211, 0.438322, 0.545944, 0.569582, 0.673944, 0.70138, 0.789556, 0.81733, 0.882076, 0.904126, 0.94379]
non_terminal_per_turn = [1, 36.0, 1260.0, 21420.0, 353430.0, 3769920.0, 38955840.0, 291950695.9908, 2115009395.3835, 11808288523.5084, 63573585617.6689, 273469809473.0522, 1131005470647.4524, 3821403760646.472, 12375577744879.377, 33169563958472.37, 84880567519773.27, 181100343278529.66, 367057606111879.7, 621582819074543.1, 993934741181799.5, 1324036811857051.2, 1654654018249388.5, 1710594484349648.8, 1644207917743448.2, 1291588645635843.5, 930625869537634.0, 537723007786683.2, 280047911573381.3, 113740858040310.55, 40500664536687.07, 10755970839056.242, 2390479324779.719, 366937758876.843, 42467386040.679596, 2816114310.0783, 95465885.78835]
term_per_turn = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 218104.0092, 3212286.3927, 53711377.251839995, 480285047.664576, 4090162894.914096, 25401311928.3924, 142842540573.5328, 644715176806.9584, 2612988261185.1875, 8931301795043.328, 26927595524431.6, 70566847443358.27, 161869090414304.47, 326725041967104.25, 572999767160974.5, 886300566832718.8, 1179451961615552.8, 1382686618112248.0, 1375520099515063.5, 1198130874225007.0, 868361545714209.6, 542724313761496.94, 274780131851059.72, 116726572559056.08, 38481976179807.54, 10153951732794.975, 1924671646776.7405, 263750312368.7586, 21373686057.477596, 834327101.37315]
illegal_per_turn = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2118.2237999999998, 53379.239760000004, 1217046.666528, 12074384.033712, 143412224.1552, 1068652379.9951999, 8599068713.664, 46900753942.4424, 240444740883.39706, 977203543038.723, 3608624730762.051, 10767631425952.4, 29513436406256.242, 66851740334934.44, 137074941308292.62, 234321334823598.4, 357837226665503.75, 457259035637892.8, 514939070750959.0, 482269599723107.2, 391169730309121.75, 258914719785429.72, 144909427420096.84, 64063302174576.22, 22862209315375.305, 6039364799746.417, 1163954220190.5618, 139162635032.4441, 8145342312.8385]

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

def plot_states(statespt: List[float], non_termpt: List[float], termpt: List[float], illegalpt: List[float], m:int, n:int, k:int):
    plt.title(label=f"m={m}, n={n}, k={k}")
    plt.plot(statespt)
    plt.plot(non_termpt)
    plt.plot(termpt)
    plt.plot(illegal_per_turn)
    labels = ["Possible States", "Non Terminal", "Terminal", "Illegal"]
    plt.legend(labels=labels)
    plt.ylabel("States")
    plt.xlabel("Turn")
    plt.show()


plot_proportions(nonterm_prop, term_prop, illegal_prop, 4, 9, 4)
# plot_log_states(states_per_turn, non_terminal_per_turn, term_per_turn, illegal_per_turn, 4, 9, 4)
# plot_states(states_per_turn, non_terminal_per_turn, term_per_turn, illegal_per_turn, 4, 9, 4)