import math
# import ArchiveManager as aManager
import numpy as np
from builtins import range
from random import seed
from random import random


from math import exp
from random import seed
from random import random
from csv import reader


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


class Backpropagation:

    # Initilialize neural network with layers making a amount neurons
    def initialize(self, nInputs, nHidden, nOutputs):
        network = list()
        hiddenLayer = [{'weights': [random() for i in range(nInputs + 1)]}
                       for i in range(nHidden)]
        network.append(hiddenLayer)
        outputLayer = [{'weights': [random() for i in range(nHidden + 1)]}
                       for i in range(nOutputs)]
        network.append(outputLayer)
        return network

    # Propagate forward
    def activate(self, inputs, weights):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return 1.0 / (1.0 + math.exp(-activation))

    def forwardPropagate(self, network, row):
        inputs = row
        for layer in network:
            newInputs = []
            for neuron in layer:
                activation = self.activate(inputs, neuron['weights'])
                neuron['output'] = self.transfer(activation)
                newInputs.append(neuron['output'])
            inputs = newInputs
        return inputs

    # Propagate backwards
    def transferDerivative(self, output):
        return output * (1.0 - output)

    def backwardPropagateError(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if (i != len(network) - 1):
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * \
                    self.transferDerivative(neuron['output'])

    # For train network
    def updateWeights(self, network, row, learningRate, nOutputs):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += learningRate * \
                        neuron['delta'] * inputs[j]
                neuron['weights'][-1] += learningRate * neuron['delta']

    def updateLearningRate(self, learningRate, decay, epoch):
        return learningRate * 1 / (1 + decay * epoch)

    def trainingNetwork(self, network, train, learningRate, nEpochs, nOutputs, expectedError):
        for epoch in range(nEpochs):
            sum_error = 0
            for row in train:
                outputs = self.forwardPropagate(network, row)
                # expected = [0 for i in range(nOutputs)]
                expected = self.getExpected(row, nOutputs)
                sum_error += sum([(expected[i] - outputs[i]) **
                                  2 for i in range(len(expected))])
                self.backwardPropagateError(network, expected)
                self.updateWeights(network, row, learningRate, nOutputs)
            print('>epoch=%d, lrate=%.3f, error=%.3f' %
                  (epoch, learningRate, sum_error))

    def getExpected(self, row, nOutputs):
        expected = []
        for i in range(nOutputs):
            temp = (nOutputs - i) * - 1
            expected.append(row[temp])
        return expected

    # For predict result
    def predict(self, network, row):
        outputs = self.forwardPropagate(network, row)
        return outputs


# trainingSet = aManager.readtable('archives/training.dat')
# trainingSet = np.matrix(trainingSet)
# trainingSet = np.asfarray(trainingSet, int)
filename = 'data/dummy_data_reg.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
trainingSet = dataset


print('######################### MACHINE LEARNING WORK - BACKPROPAGATION #########################')
print('Student: Yury Alencar Lima')
print('Registration: 161150703\n')

nOutputs = 1
nEpochs = 150
nHiddenLayer = 10
learningRate = 1
expectedError = 0

seed(1)
backpropagation = Backpropagation()
nInputs = len(trainingSet[0]) - nOutputs
network = backpropagation.initialize(nInputs, nHiddenLayer, nOutputs)
backpropagation.trainingNetwork(
    network, trainingSet, learningRate, nEpochs, nOutputs, expectedError)

# input('\nPress enter to view Result...')

testSet = [[0.467, 0.069, 0.464, 0.467, 0.467, 0.467, 0.653,
            0.038, 0.309, 0.686, 0.001, 0.313, 0.512, 0.149, 0.339, 50]]

print('\n################################ BACKPROPAGATION - RESULT #################################')
for row in testSet:
    prediction = backpropagation.predict(network, row)
    # print('Input =', (row), 'Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))
    print('Expected = ', backpropagation.getExpected(
        row, nOutputs), 'Result =', (prediction))
