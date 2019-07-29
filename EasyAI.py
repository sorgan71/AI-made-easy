# example of loading the mnist dataset
from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np
import random
import keras

# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
# plot first few images
for i in range(9):
# define subplot
pyplot.subplot(330 + 1 + i)
# plot raw pixel data
pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()
trainX = trainX.reshape(trainX.shape[0], 1, 784) / 255
testX = testX.reshape(testX.shape[0], 1, 784) / 255
trainy = keras.utils.to_categorical(trainy)
testy = keras.utils.to_categorical(testy)
# one = [1]
num_ones = trainX.shape[0]
ones = np.repeat(1, num_ones)
ones.shape = (num_ones, 1, 1)
trainX = np.append(trainX, ones, axis=2)
num_ones = testX.shape[0]
ones = np.repeat(1, num_ones)
ones.shape = (num_ones, 1, 1)
testX = np.append(testX, ones, axis=2)
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
print(trainX[0])

b = random.randint(0, 100) / 100.0
layers = [784, 784, 100, 28, 100, 784, 10]

weights = []

for i in range(len(layers) - 1):
    weightAdd = []
    for j in range(layers[i]):
        weightAddAdd = []
        for e in range(layers[i + 1]):
            weightAddAdd.append(random.randint(0, 100) / 100.0)
        weightAddAdd.append(b)
        weightAdd.append(weightAddAdd)
    weightAddAdd = []
    for e in range(layers[i + 1]):
        weightAddAdd.append(0)
    weightAddAdd.append(1)
    weightAdd.append(weightAddAdd)
    weights.append(weightAdd)

print(weights[4][100])


def layer(weights, inputs):
    return np.dot(np.transpose(weights), np.transpose(inputs))


def ReLUActivation(inputs):
    output = []
    for i in range(len(inputs)):
        outputAdd = []
        if inputs[i][0] < 0:
            outputAdd = [0]
        else:
            thing = inputs[i][0]
            outputAdd = [thing]
        output.append(outputAdd)
    return output


def sigmoid(inputs):
    output = []
    if len(inputs):
        for i in range(len(inputs)):
            outputAdd = [1 / (1 + np.exp(-1 * inputs[i][0]))]
            output.append(outputAdd)
        return output
    else:
        return 1 / (1 + np.exp(-1 * inputs))


layerA = layer(weights[0], trainX[0])
layerActivationA = ReLUActivation(layerA)
layerB = layer(weights[1], np.transpose(layerActivationA))
layerActivationB = ReLUActivation(layerB)
layerC = layer(weights[2], np.transpose(layerActivationB))
layerActivationC = ReLUActivation(layerC)
layerD = layer(weights[3], np.transpose(layerActivationC))
layerActivationD = ReLUActivation(layerD)
layerE = layer(weights[4], np.transpose(layerActivationD))
layerActivationE = ReLUActivation(layerE)

layerF = layer(weights[5], np.transpose(layerActivationE))

layerActivationF = sigmoid(layerF)
print(layerActivationF)

summ = 0
for i in range(len(np.transpose(layerF)[0]) - 1):
    summ = summ + layerActivationF[i][0]

estimate = []
for i in range(len(np.transpose(layerF)[0]) - 1):
    estimate.append(layerActivationF[i][0] / summ)
print(estimate)

deltaL = []
for i in range(len(np.transpose(layerF)[0]) - 1):
    deltaL.append(estimate[i] - trainy[0])
    deltaSigmoid = sigmoid(layerF[i][0]) * (1 - sigmoid(layerF[i][0]))
    deltaL[i] = deltaL * deltaSigmoid
print(deltaL)


def gradientDescent(layer, )


def lossFunction(estimate):
    return .5 * (1 - estimate) * (1 - estimate)