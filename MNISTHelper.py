# pylint: disable=C0111,C0103,C0303,C0301,E1101,W0621
import struct
import numpy as np
from FeatureExtractor import FeatureExtractor

class MNISTHelper(object):

    @staticmethod
    def readDatabase(mode):
        if mode is 0:
            # read training set
            imagesPath = "train-images-idx3-ubyte"
            labelsPath = "train-labels-idx1-ubyte"
        else:
            imagesPath = "t10k-images-idx3-ubyte"
            labelsPath = "t10k-labels-idx1-ubyte"

        dataLabels = []
        with open(labelsPath, "rb") as f:
            _ = struct.unpack(">I", f.read(4))[0]
            size = struct.unpack(">I", f.read(4))[0]
            for _ in xrange(size):
                label = np.zeros((10, 1))
                index = struct.unpack(">B", f.read(1))[0]
                label[index] = 1
                dataLabels.append(label)
        
        dataImages = []
        with open(imagesPath, "rb") as f:
            _ = struct.unpack(">I", f.read(4))[0]
            size = struct.unpack(">I", f.read(4))[0]
            row = struct.unpack(">I", f.read(4))[0]
            column = struct.unpack(">I", f.read(4))[0]
            for _ in xrange(size):
                image = np.zeros((row*column, 1))
                for r in xrange(row):
                    for c in xrange(column):
                        brighness = struct.unpack(">B", f.read(1))[0]
                        image[c + r*column, 0] = brighness
                dataImages.append(image)
    
        ret = []
        for img, lbl in zip(dataImages, dataLabels):
            # ret.append([img, lbl])
            ret.append([img*(1/255.0), lbl])
            
        return ret

    @staticmethod
    def readNomalizeDataset(mode):
        if mode is 0:
            # read training set
            imagesPath = "train-images-idx3-ubyte-nomalized"
            labelsPath = "train-labels-idx1-ubyte"
        else:
            imagesPath = "t10k-images-idx3-ubyte-nomalized"
            labelsPath = "t10k-labels-idx1-ubyte"

        dataLabels = []
        with open(labelsPath, "rb") as f:
            _ = struct.unpack(">I", f.read(4))[0]
            size = struct.unpack(">I", f.read(4))[0]
            for _ in xrange(size):
                label = np.zeros((10, 1))
                index = struct.unpack(">B", f.read(1))[0]
                label[index] = 1
                dataLabels.append(label)
        
        dataImages = []
        with open(imagesPath, "rb") as f:
            _ = struct.unpack(">I", f.read(4))[0]
            size = struct.unpack(">I", f.read(4))[0]
            for _ in xrange(size):
                vector = np.zeros((7, 1))
                for i in xrange(7):
                    value = struct.unpack(">d", f.read(8))[0]
                    vector[i, 0] = value
                dataImages.append(vector)
    
        ret = []
        for vec, lbl in zip(dataImages, dataLabels):
            ret.append([vec / 100, lbl])
            
        return ret
        
    @staticmethod
    def writeNomalize(mode):
        if mode is 0:
            # read training set
            imagesPath = "train-images-idx3-ubyte-nomalized"
        else:
            imagesPath = "t10k-images-idx3-ubyte-nomalized"

        dataset = MNISTHelper.readDatabase(mode)

        lenght = len(dataset)

        with open(imagesPath, "w") as f:
            f.write(struct.pack(">I", 2051))
            f.write(struct.pack(">I", len(dataset)))
            count = 0
            for data in dataset:
                img = data[0]*255
                vectora = FeatureExtractor.fv(img)
                for value in vectora:
                    f.write(struct.pack(">d", value))
                print "{} / {}".format(count, lenght)
                count = count + 1
