from typing import List

import matplotlib.pyplot as plt



nonterm_prop = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9999997333333334, 0.9999994666666666, 0.9999984, 0.9999972, 0.9999914666666667, 0.9999870666666667, 0.9999716, 0.9999552, 0.9999161333333333, 0.9998782, 0.9997937333333333, 0.9997026666666666, 0.9995271333333333, 0.9993555333333334, 0.9990318, 0.9987114, 0.9981572, 0.9975958, 0.9966674, 0.9957362666666667, 0.9942578666666667, 0.9927759333333334, 0.9904981333333334, 0.9882227333333333, 0.9847954, 0.9813702, 0.9763491333333333, 0.9714288666666666, 0.9643291333333334, 0.9572772666666667, 0.9474640666666667, 0.9377757333333333, 0.9245768, 0.9115551333333334, 0.8941226666666666, 0.8769863333333333, 0.8545182]
term_prop = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6666666666666667e-07, 2.6666666666666667e-07, 1.3333333333333334e-06, 1.4666666666666667e-06, 7.066666666666667e-06, 5.866666666666667e-06, 2.2533333333333333e-05, 2.2266666666666668e-05, 6.16e-05, 6.02e-05, 0.00014606666666666668, 0.00015126666666666667, 0.0003216, 0.0003228666666666667, 0.0006452666666666667, 0.0006431333333333333, 0.0011992666666666666, 0.001204, 0.002127, 0.0021339333333333333, 0.0036039333333333333, 0.003612933333333333, 0.005875866666666667, 0.0058798, 0.0092888, 0.009282666666666666, 0.014277133333333334, 0.014151066666666667, 0.021298133333333334, 0.021092, 0.030931866666666665, 0.030547333333333333, 0.04377586666666667, 0.0430886, 0.060490866666666664, 0.059299333333333336, 0.0815916]
illegal_prop = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6666666666666667e-07, 2.6666666666666667e-07, 1.3333333333333334e-06, 1.4666666666666667e-06, 7.066666666666667e-06, 5.866666666666667e-06, 2.2533333333333333e-05, 2.2266666666666668e-05, 6.16e-05, 6.02e-05, 0.00014606666666666668, 0.00015126666666666667, 0.0003216, 0.00032293333333333334, 0.0006454666666666667, 0.0006435333333333333, 0.0012002, 0.0012056, 0.0021298, 0.0021382, 0.0036111333333333335, 0.003626, 0.005897466666666667, 0.0059158, 0.009347133333333334, 0.009373733333333334, 0.014420066666666667, 0.014372733333333334, 0.021630733333333332, 0.021604066666666668, 0.03167693333333333, 0.03164733333333333, 0.045356266666666666, 0.04538646666666667, 0.06371433333333333, 0.0638902]

states_per_turn = [1, 64.0, 4032.0, 124992.0, 3812256.0, 76245120.0, 1499487360.0, 21742566720.0, 309831575760.0, 3470113648512.0, 38171250133632.0, 343541251202688.0, 3034614385623744.0, 2.2542849721776384e+16, 1.642407622586565e+17, 1.0265047641166031e+18, 6.287341680214194e+18, 3.353248896114237e+19, 1.7511410901929905e+20, 8.055249014887757e+20, 3.6248620566994905e+21, 1.4499448226797962e+22, 5.667966125021021e+22, 1.9837881437573574e+23, 6.777942824504305e+23, 2.0855208690782475e+24, 6.256562607234742e+24, 1.6982098505351444e+25, 4.4881260335571675e+25, 1.0771502480537202e+26, 2.5133505787920137e+26, 5.340869979933029e+26, 1.1015544333611873e+27, 2.0735142275034113e+27, 3.781114179565044e+27, 6.301856965941741e+27, 1.0152991778461693e+28, 1.4962303673522495e+28, 2.126222100974249e+28, 2.764088731266524e+28, 3.455110914083155e+28, 3.9486981875236057e+28, 4.324764681573473e+28, 4.324764681573473e+28, 4.128184468774679e+28, 3.5897256250214596e+28, 2.9654255163220753e+28, 2.2240691372415565e+28, 1.5753823055461025e+28, 1.0082446755495056e+28, 6.049468053297034e+27, 3.257405874852249e+27, 1.6287029374261244e+27, 7.238679721893887e+26, 2.9490917385493613e+26, 1.0532470494819147e+26, 3.3854369447632973e+25, 9.339136399347027e+24, 2.2542743032906618e+24, 4.5085486065813234e+23, 7.5142476776355396e+22, 9.695803455013598e+21, 9.383035601626063e+20, 5.8643972510162895e+19, 1.8326241409425905e+18]

non_terminal_per_turn = [1.0, 64.0, 4032.0, 124992.0, 3812256.0, 76245120.0, 1499487360.0, 21742566720.0, 309831575760.0, 3470113648512.0, 38171250133632.0, 343541251202688.0, 3034614385623744.0, 2.2542849721776384e+16, 1.642407622586565e+17, 1.0265047641166031e+18, 6.287341680214194e+18, 3.353248672564311e+19, 1.7511408567075117e+20, 8.055245792788151e+20, 3.6248598817822563e+21, 1.4499433727349735e+22, 5.667957812004038e+22, 1.9837832504132696e+23, 6.777920683224412e+23, 2.0855073827099608e+24, 6.25650546396293e+24, 1.6981842641733963e+25, 4.488033578160876e+25, 1.0771141276154022e+26, 2.5132334566550418e+26, 5.340480808540492e+26, 1.1014428826155689e+27, 2.07320955914625e+27, 3.780372577037292e+27, 6.300128576637882e+27, 1.0149411156694488e+28, 1.4955014039172755e+28, 2.124905969493746e+28, 2.7618155446939304e+28, 3.4515714984627678e+28, 3.9433721834082737e+28, 4.3175613535198434e+28, 4.315567637001638e+28, 4.117426970473649e+28, 3.5779338549733473e+28, 2.953616401135943e+28, 2.212988824799819e+28, 1.5659640399706256e+28, 1.0007854797908553e+28, 5.996266208151249e+27, 3.222425678964169e+27, 1.608058693373465e+27, 7.127822237424971e+26, 2.8961897513065675e+26, 1.0306119980073788e+26, 3.30067260023611e+25, 9.061389860221354e+24, 2.1765524858564812e+24, 4.325075625595753e+23, 7.162426593769882e+22, 9.164772702812164e+21, 8.795411111895802e+20, 5.4376450630598345e+19, 1.6809064418364908e+18]
term_per_turn = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2235499264076.158, 11674273934619.938, 268508300496258.56, 966629881786530.8, 1.0632928699651838e+16, 4.1565084916820824e+16, 3.4385661158460864e+17, 1.0392845664239935e+18, 1.028856962078602e+19, 2.6277562950385914e+19, 1.8453880375815235e+20, 4.36844267266231e+20, 2.563617590367854e+21, 5.730439319645791e+21, 2.670434989966515e+22, 5.639958698809279e+22, 1.983661944311597e+23, 3.796238636283304e+23, 1.0952627406806745e+24, 1.815354929988951e+24, 4.612379479091201e+24, 6.602628364225368e+24, 1.4139235223338693e+25, 1.7699381509209975e+25, 3.3005851916780644e+25, 3.5846532857335324e+25, 5.606336615546411e+25, 5.397738799071852e+25, 7.088511534209042e+25, 5.942515039675018e+25, 6.612009273743331e+25, 4.722786101053143e+25, 4.425051448363373e+25, 2.655837464758464e+25, 2.0605698083140357e+25, 1.028829929532917e+25, 6.479052671878342e+24, 2.630963382406247e+24, 1.3137431314067784e+24, 4.208549513933414e+23, 1.5989473168412716e+23, 3.854132776336045e+22, 1.0475312117484599e+22, 1.735430529645283e+21, 3.006280819261516e+20, 2.883256711810063e+19, 2.394476403169792e+18, 7.39355105171306e+16]
illegal_per_turn = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11674273934619.938, 53701660099251.71, 1208287352233163.5, 3866519527146123.0, 4.1565084916820824e+16, 1.4547779720887286e+17, 1.1748434229140795e+18, 3.1977986659199795e+18, 3.086570886235806e+19, 7.132481372247605e+19, 4.8770969564654554e+20, 1.0484262414389544e+21, 5.981774377524993e+21, 1.2212789354113526e+22, 5.515115863028344e+22, 1.0630216273000823e+23, 3.6197866412369354e+23, 6.331265631782802e+23, 1.7652668372152066e+24, 2.6772548706489586e+24, 6.558686440805233e+24, 8.592630502597201e+24, 1.7694774694657867e+25, 2.0254189236537747e+25, 3.618674767895244e+25, 3.5907079562877354e+25, 5.359759501959125e+25, 4.703258513903117e+25, 5.866600146457172e+25, 4.468303167994103e+25, 4.695479474423676e+25, 3.0341443102869792e+25, 2.664347049820102e+25, 1.4374497804939661e+25, 1.035594475733027e+25, 4.6066957750132697e+24, 2.6592353418731395e+24, 9.497620160468183e+23, 4.2678849387852985e+23, 1.1785180744154671e+23, 3.918048967082012e+22, 7.871985981072411e+21, 1.7827803090112907e+21, 2.304026702752821e+20, 2.992988185492546e+19, 1.873045476394762e+18, 7.77821885889691e+16]

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


# plot_proportions(nonterm_prop, term_prop, illegal_prop, 4, 4, 4)
# plot_log_states(states_per_turn, non_terminal_per_turn, term_per_turn, illegal_per_turn, 4, 9, 4)
plot_states(states_per_turn, non_terminal_per_turn, term_per_turn, illegal_per_turn, 8, 8, 8)