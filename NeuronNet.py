# pylint: disable=C0111,C0103,C0303,C0301,E1101,W0621
import random
import numpy as np

class Layer(object):

    def __init__(self, size):
        """Example [2, 3]"""
        self.size = size
        self.biases = np.random.randn(size[1], 1)
        self.weights = np.random.randn(size[1], size[0])

    def Z(self, x):
        return np.dot(self.weights, x) + self.biases

    def activation(self, z):
        pass
    
    def activationDev(self, z):
        pass

    def feed(self, x):
        pass
    
    def getOutputGradient(self, z, y):
        pass
    
    def getBackwardGradient(self, z, nW, nSig):
        pass

    def getWeightDelta(self, nSig, lA):
        pass

    def updateBiases(self, nSig, rate, dataSize):
        self.biases = self.biases - (rate / dataSize) * nSig
    
    def updateWeights(self, nSig, rate, l2Rate, datasetSize, dataSize):
        self.weights = (1 - rate*l2Rate / datasetSize)*self.weights - (rate / dataSize) * nSig

class LogisticLayer(Layer):

    def __init__(self, size):
        """Example [2, 3]"""
        Layer.__init__(self, size)

    def activation(self, z):
        return 1/(1 + np.exp(-z))
    
    def activationDev(self, z):
        return self.activation(z)*(1 - self.activation(z))

    def feed(self, x):
        return self.activation(self.Z(x))
    
    def getOutputGradient(self, z, y):
        """cross-entropy"""
        return self.activation(z) - y
    
    def getBackwardGradient(self, z, nW, nSig):
        return np.dot(nW.transpose(), nSig)*self.activationDev(z)

    def getWeightDelta(self, nSig, lA):
        return np.dot(nSig, lA.transpose())
    
class LinearLayer(Layer):

    def __init__(self, size):
        """Example [2, 3]"""
        Layer.__init__(self, size)

    def activation(self, z):
        return z
    
    def activationDev(self, z):
        return self.activation(z)*(1 - self.activation(z))

    def feed(self, x):
        return self.activation(self.Z(x))
    
    def getOutputGradient(self, z, y):
        """cross-entropy"""
        a = self.activation(z)
        return (a - y) / (a*(1 - a))
    
    def getBackwardGradient(self, z, nW, nSig):
        return np.dot(nW.transpose(), nSig)

    def getWeightDelta(self, nSig, lA):
        return np.dot(nSig, lA.transpose())

class Network(object):

    def __init__(self, layers):
        self.layers = layers

    def printNetwork(self):
        self.printBiases()
        self.printWeights()

    def printBiases(self):
        print "\nBiases\n"
        for layer in self.layers:
            print layer.biases
    
    def printWeights(self):
        print "\nWeights\n"
        for layer in self.layers:
            print layer.weights

    def feedForward(self, x):
        feedZ = []
        feedA = [x]
        for layer in self.layers: 
            z = layer.Z(x)
            x = layer.activation(z)
            feedZ.append(z)
            feedA.append(x)
        return feedZ, feedA
    
    def backpropagation(self, x, y):
        feedZ, feedA = self.feedForward(x)
        
        outputBiases = []
        outputWeights = []

        lastLayer = self.layers[-1]
        lastBias = lastLayer.getOutputGradient(feedZ[-1], y)
        outputBiases.append(lastBias)
        lastWeight = lastLayer.getWeightDelta(lastBias, feedA[-2])
        outputWeights.append(lastWeight)

        for layer, nLayer, i in zip(reversed(self.layers[:-1]), reversed(self.layers[1:]), xrange(len(self.layers[:-1]))):
            backwardGradient = layer.getBackwardGradient(feedZ[-i - 2], nLayer.weights, outputBiases[-1])
            backwardWeight = layer.getWeightDelta(backwardGradient, feedA[-i - 3])

            outputBiases.append(backwardGradient)
            outputWeights.append(backwardWeight)
        
        outputBiases.reverse()
        outputWeights.reverse()
        return outputBiases, outputWeights

    def learnFromData(self, datas, rate):
        if len(datas) is 0:
            return
        sumBiases = []
        sumWeights = []

        for layer in self.layers:
            sumBiases.append(np.zeros(layer.biases.shape))
            sumWeights.append(np.zeros(layer.weights.shape))
        
        for data in datas:
            x = data[0]
            y = data[1]
            outBias, outWeight = self.backpropagation(x, y)
            sumBiases = [sb + ob for sb, ob in zip(sumBiases, outBias)]
            sumWeights = [sw + ow for sw, ow in zip(sumWeights, outWeight)]

        # update self layers
        for layer, sumb, sumw in zip(self.layers, sumBiases, sumWeights):
            layer.updateBiases(sumb, rate, len(datas))
            layer.updateWeights(sumw, rate, 0.1, 60000, len(datas))

    def SGD(self, dataset, epoch, batchsize, rate, testset=None):
        sizeSpread = int(len(dataset) / batchsize) + 1
        step = 1
        for _ in xrange(epoch):
            random.shuffle(dataset)
            batchs = [dataset[i*batchsize:i*batchsize + batchsize] for i in xrange(sizeSpread)]
            for batch in batchs:
                self.learnFromData(batch, rate)
            if testset:
                print "Epoch {0}: {1}/{2}".format(step, self.validateNetwork(testset), len(testset))
            step = step + 1
                
    def validateNetwork(self, testset):
        i = 0
        for data in testset:
            _, feedA = self.feedForward(data[0])
            if np.argmax(feedA[-1]) == np.argmax(data[1]):
                i = i + 1
        return i
    
    def buildSampleDataset(self, num, size):
        ret = []
        for _ in xrange(num):
            x = np.random.randn(size[0], 1)
            y = np.random.randn(size[-1], 1)
            ret.append([x, y])
        return ret
