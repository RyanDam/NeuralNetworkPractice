# pylint: disable=C0111,C0103,C0303,C0301,E1101,W0621

import numpy as np
import cv2

class FeatureExtractor(object):

    @staticmethod
    def m(img, p, q):
        M = img.shape[0]
        N = img.shape[1]
        image = img
        val = 0.0
        for x in xrange(M):
            for y in xrange(N):
                val = val + pow(x, p)*pow(y, q)*image[x, y]
        return val
    
    @staticmethod
    def moun(img, p, q):
        M = img.shape[0]
        N = img.shape[1]
        image = img
        val = 0.0
        m10 = FeatureExtractor.m(img, 1, 0)
        m01 = FeatureExtractor.m(img, 0, 1)
        m00 = FeatureExtractor.m(img, 0, 0)
        xAvg = m10 / m00
        yAvg = m01 / m00
        for x in xrange(M):
            for y in xrange(N):
                val = val + pow(x - xAvg, p)*pow(y - yAvg, q)*image[x, y]
        return val

    @staticmethod
    def eta(img, p, q):
        gamma = ((p + q) / 2) + 1
        mounPQ = FeatureExtractor.moun(img, p, q)
        moun00 = FeatureExtractor.moun(img, 0, 0)
        return mounPQ / pow(moun00, gamma)

    @staticmethod
    def fv(img):
        eta20 = FeatureExtractor.eta(img, 2, 0)
        eta02 = FeatureExtractor.eta(img, 0, 2)
        eta11 = FeatureExtractor.eta(img, 1, 1)
        eta30 = FeatureExtractor.eta(img, 3, 0)
        eta03 = FeatureExtractor.eta(img, 0, 3)
        eta12 = FeatureExtractor.eta(img, 1, 2)
        eta21 = FeatureExtractor.eta(img, 2, 1)

        phi1 = eta20 + eta02
        phi2 = pow(eta20 - eta02, 2) + 4*pow(eta11, 2)
        phi3 = pow(eta30 - 3*eta12, 2) + pow(3*eta21 - eta03, 2)
        phi4 = pow(eta30 + eta12, 2) + pow(eta21 + eta03, 2)
        phi5 = (eta30 - 3*eta12)*(eta30 + eta12)*(pow(eta30 + eta12, 2) - 3*pow(eta21 + eta03, 2)) + (3*eta21 - eta03)*(eta21 + eta03)*(3*pow(eta30 + eta12, 2) - pow(eta21 + eta03, 2))
        phi6 = (eta20 - eta02)*(pow(eta30 + eta12, 2) - pow(eta21 + eta03, 2)) + 4*eta11*(eta30 + eta12)*(eta21 + eta03)
        phi7 = (3*eta21 - eta03)*(eta30 + eta12)*(pow(eta30 + eta12, 2) - 3*pow(eta21 + eta03, 2)) + (3*eta12 - eta30)*(eta21 + eta03)*(3*pow(eta30 + eta12, 2) - pow(eta21 + eta03, 2))

        ret = np.array([phi1, phi2, phi3, phi4, phi5, phi6, phi7]).reshape(7, 1)

        return ret

    @staticmethod
    def example():
        graya = cv2.imread("a.png", 0)
        grayb = cv2.imread("b.png", 0)
        grayc = cv2.imread("c.png", 0)
        grayd = cv2.imread("d.png", 0)
        
        vectora = FeatureExtractor.fv(graya)
        vectorb = FeatureExtractor.fv(grayb)
        vectorc = FeatureExtractor.fv(grayc)
        vectord = FeatureExtractor.fv(grayd)

        print "a"
        print vectora.reshape(1, 7)
        print "b"
        print vectorb.reshape(1, 7)
        print "c"
        print vectorc.reshape(1, 7)
        print "d"
        print vectord.reshape(1, 7)
