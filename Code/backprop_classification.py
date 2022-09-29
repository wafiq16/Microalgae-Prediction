from math import exp
from random import seed
from random import random
from csv import reader
# Initialize a network
# Load a CSV file


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column


def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]}
                    for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]}
                    for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output


def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) **
                             2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    print(outputs)
    return outputs.index(max(outputs))


filename = 'data/dummy_data_cla.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
print(n_inputs)
print(n_outputs)
n_epoch = 150
LR = 0.1
n_hidden = 17
network = initialize_network(n_inputs, n_hidden, n_outputs)
train_network(network, dataset, LR, n_epoch, n_outputs)

for layer in network:
    print(layer)

# Test making predictions with the network
dataset1 = [[0.4355509072410918, 0.059687488587916525, 0.5047616041709917, 0.527393047285987, 0.018490538244564023, 0.45411641446944895, 0.536745275332369, 0.01754347478800922, 0.44571124987962174, 0.5839919752708758, 0.0671844369652191, 0.34882358776390515, 0.6275302615037311, 0.057905181772247925, 0.31456455672402095, 0],
            [0.30307749590764055, 0.04883918730676659, 0.6480833167855928, 0.30234776639566296, 0.014064277307867207, 0.6835879562964698, 0.378861492991622, 0.16906873147153875,
                0.4520697755368392, 0.3066139091679242, 0.107880575210088, 0.5855055156219878, 0.3817573440332732, 0.04826324645071224, 0.5699794095160146, 2],
            [0.3725041646045023, 0.6215313105256031, 0.005964524869894641, 0.3164142249943299, 0.6832931309306132, 0.0002926440750568847, 0.4680993644887926, 0.48900287495739286,
                0.04289776055381461, 0.32234111231290974, 0.6746851604642844, 0.002973727222805878, 0.31684539096839104, 0.5525481245480538, 0.13060648448355514, 1],
            [0.31358535039535906, 0.5693090021211102, 0.11710564748353074, 0.31766312301713995, 0.6041448963121506, 0.07819198067070948, 0.5537547379162276, 0.443278689939098,
                0.002966572144674372, 0.30670719410655056, 0.6717154353947146, 0.021577370498734892, 0.5047512699483254, 0.4495227971968417, 0.04572593285483294, 1],
            [0.40961881026343905, 0.0037701197160500046, 0.5866110700205109, 0.3098498405015326, 0.07722401222860924, 0.6129261472698582, 0.3676532244901298, 0.028044173248719304,
                0.6043026022611508, 0.3008260549191296, 0.009937517905507329, 0.689236427175363, 0.34784034445724954, 0.20196167195624184, 0.4501979835865086, 2],
            [0.43634013828160373, 0.4449007584410895, 0.11875910327730675, 0.47205326205282144, 0.4686769545505372, 0.0592697833966413, 0.3264201170333436, 0.6619749106026167, 0.011604972364039703, 0.30371696346404, 0.6837502141856147, 0.012532822350345401, 0.41243912874089395, 0.5867340209491718, 0.0008268503099342609, 1]]

network = [[{'weights': [-0.062351296734103716, 0.22770950791299963, 0.29289126997312165, 0.6678072553981319, 0.4408902526148873, 0.7202703609075148, 0.26774052582332697, 0.691083512959838, 0.3406383634798136, 0.5039806277437329, -0.011987598977689765, 1.1632026843420404, 0.44440446231666403, 0.852342803483097, 0.37582197759832403, 0.17612307856933287]},
           {'weights': [1.0424557737619204, -1.225866937937812, 0.6521342956977668, 0.49873446147695893, -1.3087600541064446, 1.0621215646371844, 1.095795407417641, -1.198805936036034, 0.8455489553315659,
                        1.006929458108246, -0.8676616167588851, 1.0689273227142626, 0.31984384553250406, -0.4205382772792435, 1.3303312421219258, -0.6713742826617177]},
           {'weights': [-0.06448032248998213, -0.22141780875134434, 1.0212855019889786, 0.33690002679793574, -1.010991141306388, 1.1072944909163926, 0.12732667002517542, -0.2146256192086783, 0.7199701630150077,
                        0.8982526446899591, -0.9612204138948354, 1.0703073396763623, 0.7785852072622365, -1.0426648793866986, 0.839911409946096, -0.7327773849325865]},
           {'weights': [-0.07603474361901098, 0.5039335650945058, 0.67462896186547, -0.17903300220407517, 0.8498091370546208, 0.5166925646272963, -0.2591145950047142, 0.16104510654069965, -0.02775626179657725, -
                        0.6049497805719908, 0.9379882448051788, 0.6724436521751226, -0.3902437245938031, 0.17234722288453427, 0.31192183108545624, 0.03926042971113972]},
           {'weights': [0.8763102618801186, 0.2534839370765042, 0.7021231892483405, 0.5952681797866348, 0.5220353142621754, 1.013392803441934, 0.41650743285816677, 0.8391676493430352, 0.549329628650698,
                        0.762108286411712, 0.7456480037253159, 0.44768057262180005, 0.0024143454784826303, 0.7277955473514709, 0.6120762798130224, 0.9938589006445212]},
           {'weights': [0.8163567936256053, 0.8304654648252098, 0.5767645164717772, 0.24038068059087275, 0.8052886254061872, -0.1959421832799169, 0.7153362502662716, 0.40965302664194636, 0.13948451577966173,
                        0.2592507894685147, 0.26001628044918207, 0.038624752100450424, 0.924796316679603, 0.8911461176849873, 0.6849387970897041, 0.6656679136004796]},
           {'weights': [1.194875416420653, -0.9056079906416528, -0.43397875201238906, 2.6767768481735965, -0.969259586034518, -0.7530581643880533, 1.3916029586636762, -0.38572864216935193, 0.08967700887701022,
                        3.4904514805162945, -1.2530930944379914, -1.174371820691921, 1.4164590443911436, -0.17101264035992286, -0.7282855785675132, -1.3673343816469847]},
           {'weights': [0.6040314187468587, 0.42772811366429664, 0.1265775474059733, 0.3025733601412965, 0.7302473870030964, 0.7258483166062997, 0.345659368849079, 0.2604128549049954, 0.9398118246817104,
                        0.3847690601680804, 0.12131706631987407, 0.7318355045327578, 0.3335227661030074, 0.9459789148118047, 0.3872381356297122, 0.04685784336154524]},
           {'weights': [0.07369198970241725, 0.36155810311257874, 0.06395747883533884, 0.811915036146108, 0.6360134736542542, 0.009667965289933087, -0.022127478151147403, 0.0640958785100455, 0.7276198352845875,
                        0.07607120754361546, 0.8333955755635474, 0.6232363538499507, -0.010719427119346106, 0.9984674473483306, -0.012019903276034919, 0.2787111685934474]},
           {'weights': [-0.062381860097682915, 0.5126698470508255, -0.23516740269801306, -0.5586819823385998, 0.6439322746428101, 0.6517181316317802, -0.3063448438979092, 0.9374745762351131, 0.06229477353762171, -
                        0.6924239909909989, 0.4579062539270207, 0.8309751766469857, -0.16561739988886318, 0.40406539365155475, -0.03399156958946488, 0.22336831744781263]},
           {'weights': [0.6944211579638714, 1.0166210203416788, -0.5659980716751893, 2.261222134721949, 1.3636897728250241, -1.4511008335611284, 0.993246165210918, 0.5133574851812747, -0.4264080909628446,
                        2.8302599828900803, 1.1944798711231983, -1.7533079905424505, 0.953732207980008, 0.9390140706115628, -0.7505186064909354, -0.3747404461393493]},
           {'weights': [0.5175655033206026, 0.9092575210707351, 0.5575928474400713, 0.7579341806486066, 0.48174415676611076, 0.57163990889599, 0.018524987482352706, 0.924356885823573, 1.0211793994764864,
                        0.5221266559245071, 0.5269341458845708, 0.21419902785331968, 0.9504790805585918, 0.4524423595607609, 0.515812786486739, 0.6620364031080843]},
           {'weights': [0.46314284473393696, 0.32716469288215944, 0.35048056265611094, -0.12488635857797159, 0.5759927156104362, 0.39962375344704376, 0.5765285628895495, 0.9114677358731244, 0.5664237056696116, -
                        0.1740983509332407, 1.0283804725459793, 0.6050057899578346, -0.16368667620695684, 1.0160941362104894, -0.09910502506462249, 0.5780313737612313]},
           {'weights': [0.35046548293943436, 0.8628530318541605, 0.8909045640924015, 0.35283307377720763, 0.8607848960985105, 0.4305633051844097, 0.8471937182904802, 0.5335832606335259, 0.2024366052680155,
                        0.2582003633761979, 0.34586456644621955, 0.443773252095974, 0.6172808404362182, 0.04805101439609455, 0.2763258049021524, 0.9672800626231977]},
           {'weights': [1.0529946209472782, 0.23219616388736472, -0.5534256378320126, 1.9933281360768207, 0.23133556995737917, -1.1518430970367821, 0.6859471333352514, 0.15452634573609064, -0.5571693349392521,
                        1.7009218070190137, -0.46506559769020905, -0.5140143992818135, 0.9733370218738336, -0.13737015954746515, -0.05002778480625636, -0.6740330455640434]},
           {'weights': [0.743674939280247, 0.21196014262478732, 1.04588110899908, 0.49063983175898074, 0.5405185346301952, 0.5608485573110186, 0.7141142791659548, 0.24678314866930573, 0.4352099791747783,
                        0.3989176911610138, 0.44991314854633935, 0.25305124128971407, 0.2885661553215256, 0.3922334469597779, 0.74049230088506, 0.6772194128016222]},
           {'weights': [0.14061143019797448, 0.25762142033069435, 0.5762160915608268, 0.5962245832338942, 0.07210339325493666, 0.5176917757519072, 0.12902453508682596, 0.7505050131910601, 0.15332043652201602, 0.2853071629200838, 0.7153398198695469, 0.4399041757556406, 0.6607648856287367, 0.5612189051867997, 0.203735252456837, 0.34735319861258657]}],
           [{'weights': [-0.6012058561761543, 2.488256339732563, 1.0132011213583167, -1.2561205929113857, -0.9633741201717945, -0.7137281536474082, 5.48077185033174, -0.3605637395339739, -0.411724572417199, -1.2313147558556898, 2.5220136123251797, -0.7086070124214148, -1.0819899105351307, -0.701150420125215, 2.6366705676591935, -0.6568861473945058, -0.1158972163817127, -1.0665908057624092]},
            {'weights': [-0.23947747514852546, -4.191226981676723, -2.76858784643048, 0.25389095933494366, -0.3514722508302387, 0.7022933091296141, -2.815522782574432, -0.057402831234010215, 0.42076862137730076, 0.8646057290055023,
                         2.7093972173987995, 0.14855161457985003, 0.6716782692286546, 0.1580496564391696, -0.4711237554785721, -0.019375237338485857, -0.03812229839124696, 0.08959628722936576]},
            {'weights': [0.39882594649749253, 2.3091268693826614, 1.6866174854443767, 0.6707508103539411, 0.19861772547209683, -0.6290895149595679, -3.1478194720689627, -0.02272409934132218, -0.12481119921013203, 0.7955505833108716, -5.293034488471244, 0.28962288698942146, 0.14047671076788393, -0.23276154160734536, -2.285527617778804, 0.4979130631615336, -0.09953536745890702, 0.07747937881472072]}]]

for row in dataset1:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

# [{'weights': [-0.062351296734103716, 0.22770950791299963, 0.29289126997312165, 0.6678072553981319, 0.4408902526148873, 0.7202703609075148, 0.26774052582332697, 0.691083512959838, 0.3406383634798136, 0.5039806277437329, -0.011987598977689765, 1.1632026843420404, 0.44440446231666403, 0.852342803483097, 0.37582197759832403, 0.17612307856933287], 'output': 0.953431764142068, 'delta': -2.2738739455378115e-06},
# {'weights': [1.0424557737619204, -1.225866937937812, 0.6521342956977668, 0.49873446147695893, -1.3087600541064446, 1.0621215646371844, 1.095795407417641, -1.198805936036034, 0.8455489553315659, 1.006929458108246, -0.8676616167588851, 1.0689273227142626, 0.31984384553250406, -0.4205382772792435, 1.3303312421219258, -0.6713742826617177], 'output': 0.9809958972782418, 'delta': -7.742966894233442e-06},
# {'weights': [-0.06448032248998213, -0.22141780875134434, 1.0212855019889786, 0.33690002679793574, -1.010991141306388, 1.1072944909163926, 0.12732667002517542, -0.2146256192086783, 0.7199701630150077, 0.8982526446899591, -0.9612204138948354, 1.0703073396763623, 0.7785852072622365, -1.0426648793866986, 0.839911409946096, -0.7327773849325865], 'output': 0.960592882005418, 'delta': -1.1195997294050174e-05},
# {'weights': [-0.07603474361901098, 0.5039335650945058, 0.67462896186547, -0.17903300220407517, 0.8498091370546208, 0.5166925646272963, -0.2591145950047142, 0.16104510654069965, -0.02775626179657725, -0.6049497805719908, 0.9379882448051788, 0.6724436521751226, -0.3902437245938031, 0.17234722288453427, 0.31192183108545624, 0.03926042971113972], 'output': 0.8691379141722645, 'delta': -3.1575460255928776e-06},
# {'weights': [0.8763102618801186, 0.2534839370765042, 0.7021231892483405, 0.5952681797866348, 0.5220353142621754, 1.013392803441934, 0.41650743285816677, 0.8391676493430352, 0.549329628650698, 0.762108286411712, 0.7456480037253159, 0.44768057262180005, 0.0024143454784826303, 0.7277955473514709, 0.6120762798130224, 0.9938589006445212], 'output': 0.9888093586616093, 'delta': -7.308312488994684e-07},
# {'weights': [0.8163567936256053, 0.8304654648252098, 0.5767645164717772, 0.24038068059087275, 0.8052886254061872, -0.1959421832799169, 0.7153362502662716, 0.40965302664194636, 0.13948451577966173, 0.2592507894685147, 0.26001628044918207, 0.038624752100450424, 0.924796316679603, 0.8911461176849873, 0.6849387970897041, 0.6656679136004796], 'output': 0.9143206697407559, 'delta': 5.388217114726927e-06},
# {'weights': [1.194875416420653, -0.9056079906416528, -0.43397875201238906, 2.6767768481735965, -0.969259586034518, -0.7530581643880533, 1.3916029586636762, -0.38572864216935193, 0.08967700887701022, 3.4904514805162945, -1.2530930944379914, -1.174371820691921, 1.4164590443911436, -0.17101264035992286, -0.7282855785675132, -1.3673343816469847], 'output': 0.04062223042837429, 'delta': -1.602486756521259e-06},
# {'weights': [0.6040314187468587, 0.42772811366429664, 0.1265775474059733, 0.3025733601412965, 0.7302473870030964, 0.7258483166062997, 0.345659368849079, 0.2604128549049954, 0.9398118246817104, 0.3847690601680804, 0.12131706631987407, 0.7318355045327578, 0.3335227661030074, 0.9459789148118047, 0.3872381356297122, 0.04685784336154524], 'output': 0.9600236573249035, 'delta': -5.590478427795859e-07},
# {'weights': [0.07369198970241725, 0.36155810311257874, 0.06395747883533884, 0.811915036146108, 0.6360134736542542, 0.009667965289933087, -0.022127478151147403, 0.0640958785100455, 0.7276198352845875, 0.07607120754361546, 0.8333955755635474, 0.6232363538499507, -0.010719427119346106, 0.9984674473483306, -0.012019903276034919, 0.2787111685934474], 'output': 0.8663372691977057, 'delta': 3.970309834598144e-06},
# {'weights': [-0.062381860097682915, 0.5126698470508255, -0.23516740269801306, -0.5586819823385998, 0.6439322746428101, 0.6517181316317802, -0.3063448438979092, 0.9374745762351131, 0.06229477353762171, -0.6924239909909989, 0.4579062539270207, 0.8309751766469857, -0.16561739988886318, 0.40406539365155475, -0.03399156958946488, 0.22336831744781263], 'output': 0.811143708230598, 'delta': 4.532786063922576e-06},
# {'weights': [0.6944211579638714, 1.0166210203416788, -0.5659980716751893, 2.261222134721949, 1.3636897728250241, -1.4511008335611284, 0.993246165210918, 0.5133574851812747, -0.4264080909628446, 2.8302599828900803, 1.1944798711231983, -1.7533079905424505, 0.953732207980008, 0.9390140706115628, -0.7505186064909354, -0.3747404461393493], 'output': 0.024385491225691953, 'delta': 1.1626846199694146e-05},
# {'weights': [0.5175655033206026, 0.9092575210707351, 0.5575928474400713, 0.7579341806486066, 0.48174415676611076, 0.57163990889599, 0.018524987482352706, 0.924356885823573, 1.0211793994764864, 0.5221266559245071, 0.5269341458845708, 0.21419902785331968, 0.9504790805585918, 0.4524423595607609, 0.515812786486739, 0.6620364031080843], 'output': 0.9743423457533386, 'delta': -3.1370440084747106e-07},
# {'weights': [0.46314284473393696, 0.32716469288215944, 0.35048056265611094, -0.12488635857797159, 0.5759927156104362, 0.39962375344704376, 0.5765285628895495, 0.9114677358731244, 0.5664237056696116, -0.1740983509332407, 1.0283804725459793, 0.6050057899578346, -0.16368667620695684, 1.0160941362104894, -0.09910502506462249, 0.5780313737612313], 'output': 0.936006393568734, 'delta': 2.011039003862612e-06},
# {'weights': [0.35046548293943436, 0.8628530318541605, 0.8909045640924015, 0.35283307377720763, 0.8607848960985105, 0.4305633051844097, 0.8471937182904802, 0.5335832606335259, 0.2024366052680155, 0.2582003633761979, 0.34586456644621955, 0.443773252095974, 0.6172808404362182, 0.04805101439609455, 0.2763258049021524, 0.9672800626231977], 'output': 0.9642904292246948, 'delta': 1.3105130256236313e-07},
# {'weights': [1.0529946209472782, 0.23219616388736472, -0.5534256378320126, 1.9933281360768207, 0.23133556995737917, -1.1518430970367821, 0.6859471333352514, 0.15452634573609064, -0.5571693349392521, 1.7009218070190137, -0.46506559769020905, -0.5140143992818135, 0.9733370218738336, -0.13737015954746515, -0.05002778480625636, -0.6740330455640434], 'output': 0.07772964957716455, 'delta': 6.4481451061423465e-06},
# {'weights': [0.743674939280247, 0.21196014262478732, 1.04588110899908, 0.49063983175898074, 0.5405185346301952, 0.5608485573110186, 0.7141142791659548, 0.24678314866930573, 0.4352099791747783, 0.3989176911610138, 0.44991314854633935, 0.25305124128971407, 0.2885661553215256, 0.3922334469597779, 0.74049230088506, 0.6772194128016222], 'output': 0.9772617080160487, 'delta': -7.518319809550253e-07},
# {'weights': [0.14061143019797448, 0.25762142033069435, 0.5762160915608268, 0.5962245832338942, 0.07210339325493666, 0.5176917757519072, 0.12902453508682596, 0.7505050131910601, 0.15332043652201602, 0.2853071629200838, 0.7153398198695469, 0.4399041757556406, 0.6607648856287367, 0.5612189051867997, 0.203735252456837, 0.34735319861258657], 'output': 0.9058953971552846, 'delta': -3.3569808191386347e-07}]
# [{'weights': [-0.6012058561761543, 2.488256339732563, 1.0132011213583167, -1.2561205929113857, -0.9633741201717945, -0.7137281536474082, 5.48077185033174, -0.3605637395339739, -0.411724572417199, -1.2313147558556898, 2.5220136123251797, -0.7086070124214148, -1.0819899105351307, -0.701150420125215, 2.6366705676591935, -0.6568861473945058, -0.1158972163817127, -1.0665908057624092], 'output': 0.005162349949243621, 'delta': 2.6512281110532292e-05},
# {'weights': [-0.23947747514852546, -4.191226981676723, -2.76858784643048, 0.25389095933494366, -0.3514722508302387, 0.7022933091296141, -2.815522782574432, -0.057402831234010215, 0.42076862137730076, 0.8646057290055023, 2.7093972173987995, 0.14855161457985003, 0.6716782692286546, 0.1580496564391696, -0.4711237554785721, -0.019375237338485857, -0.03812229839124696, 0.08959628722936576], 'output': 0.009989067141508828, 'delta': 9.878473863060364e-05},
# {'weights': [0.39882594649749253, 2.3091268693826614, 1.6866174854443767, 0.6707508103539411, 0.19861772547209683, -0.6290895149595679, -3.1478194720689627, -0.02272409934132218, -0.12481119921013203, 0.7955505833108716, -5.293034488471244, 0.28962288698942146, 0.14047671076788393, -0.23276154160734536, -2.285527617778804, 0.4979130631615336, -0.09953536745890702, 0.07747937881472072], 'output': 0.9945878766444424, 'delta': -2.9132552281839207e-05}]
