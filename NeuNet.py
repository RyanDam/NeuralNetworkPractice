# pylint: disable=C0111,C0103,C0303,C0301,E1101,W0621

import random
import numpy as np
# from MNISTHelper import MNISTHelper

class NeuNet(object):
    """Implementation for neurons network"""

    def __init__(self, neuStruct):
        """example [748, 30, 10]"""
        self.numLayers = len(neuStruct)
        self.struct = neuStruct
        self.biases = [np.random.randn(y, 1) for y in neuStruct[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(neuStruct[1:], neuStruct[:-1])]
    
    def feedforward(self, x):
        zs = []
        activation = x
        activations = [x]
        for a, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + a
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)
        return zs, activations

    def backpropation(self, x, output):
        deltaBiases = [np.zeros((i, 1)) for i in self.struct[1:]]
        deltaWeights = [np.zeros((i, j)) for i, j in zip(self.struct[1:], self.struct[:-1])]
        # feed forward to save all activation layer by layers 
        # and z layer by layers
        zs, activations = self.feedforward(x)
        # calculate errors for output layer
        lastActiveDerv = self.activitionDerivation(zs[-1])
        lastDetalOutput = self.costFunctionDerivation(activations[-1], output)
        lastDeltaBias = lastDetalOutput*lastActiveDerv
        deltaBiases[-1] = lastDeltaBias
        deltaWeights[-1] = np.dot(lastDeltaBias, activations[-2].transpose())
        # calculate error for all layers
        for l in xrange(2, self.numLayers):
            nextWeight = self.weights[-l + 1]
            nextDelta = deltaBiases[-l + 1]
            aderiv = self.activitionDerivation(zs[-l])
            delta = np.dot(nextWeight.transpose(), nextDelta)*aderiv
            deltaBiases[-l] = delta
            deltaWeights[-l] = np.dot(delta, activations[-l - 1].transpose())
        return deltaBiases, deltaWeights

    def updateMiniBatch(self, dataset, rate):
        deltaBiases = [np.zeros((i, 1)) for i in self.struct[1:]]
        deltaWeights = [np.zeros((i, j)) for i, j in zip(self.struct[1:], self.struct[:-1])]
        for data in dataset:
            nablaBiases, nablaWeights = self.backpropation(data[0], data[1])
            deltaBiases = [db + nb for db, nb in zip(deltaBiases, nablaBiases)]
            deltaWeights = [dw + nw for dw, nw in zip(deltaWeights, nablaWeights)]
        self.biases = [b - (rate/len(dataset))*db for b, db in zip(self.biases, deltaBiases)]
        self.weights = [w - (rate/len(dataset))*dw for w, dw in zip(self.weights, deltaWeights)]

    def learnFromDataset(self, dataset, fold, spread, rate, testData=None):
        batchSize = int(len(dataset)/spread)
        for f in xrange(fold):
            random.shuffle(dataset)
            batchSet = [dataset[k:k+batchSize] for k in xrange(0, spread)]
            for batch in batchSet:
                self.updateMiniBatch(batch, rate)
            if testData is not None:
                print "fold {0}: {1} / {2}".format(f, self.validateData(testData), len(testData))
            
    def validateData(self, dataset):
        i = 0
        for data in dataset:
            _, active = self.feedforward(data[0])
            lastActive = active[-1]
            if np.argmax(lastActive) == np.argmax(data[1]):
                i = i + 1
        return i

    def costFunctionDerivation(self, y, output):
        return y - output
    
    def activation(self, z):
        return self.sigmoid(z)

    def activitionDerivation(self, z):
        return self.sigmoidDerivation(z)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidDerivation(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def printOut(self):
        """Print out neunet"""
        print "\nBiases"
        print self.biases
        print "\nWeights"
        print self.weights

    def buildSampleDataset(self, num):
        ret = []
        for _ in xrange(num):
            x = np.random.randn(self.struct[0], 1)
            y = np.random.randn(self.struct[-1], 1)
            ret.append([x, y])
        return ret

# net = NeuNet([784, 30, 10])
# net.printOut()

# print "Preparing data set"
# mnist = MNISTHelper()
# trainingSet = mnist.readDatabase(0) # read training set
# testSet = mnist.readDatabase(1) # read testing set

# print "Start training"
# net.learnFromDataset(trainingSet, 30, 100, 0.1, testSet)

# net.printOut()

