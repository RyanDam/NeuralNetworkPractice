# pylint: disable=C0111,C0103,C0303,C0301,E1101,W0621
# from NeuNet import NeuNet
from MNISTHelper import MNISTHelper
# import cv2
# import numpy as np
# from MNISTHelper import MNISTHelper

# net = NeuNet([784, 38, 10])

# print "start learning"
# net.learnFromDataset(dataset, 30, 6000, 3.0, testset)

from NeuronNet import Network, LogisticLayer, LinearLayer

lay1 = LogisticLayer([784, 30])
lay2 = LogisticLayer([30, 10])

net = Network([lay1, lay2])

# net.printNetwork()

print "building dataset"
dataset = MNISTHelper.readDatabase(0)
print "building testset"
testset = MNISTHelper.readDatabase(1)

# for i in xrange(20):
#     print "\n{}".format(dataset[i][0])

# sample = net.buildSampleDataset(30, [2, 4])

net.SGD(dataset, 30, 10, 0.5, testset)

# net.printNetwork()
