'''
Yousif Alnajjar - 112 TP - Node Class
Sources:
https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://github.com/Manik9/LSTMs
'''
import numpy as np

class Node:
    def __init__(self, params, cells):
        self.f, self.i, self.o, self.C, self.x, self.h = (np.zeros(cells) for _ in range(6))
        self.dXBottom = np.zeros_like(self.x)
        self.dHBottom = np.zeros_like(self.h)
        self.params = params

    def hLine(self, x, hPrev, xPrev):
        if hPrev is None:
            hPrev = np.zeros_like(self.h)
            xPrev = np.zeros_like(self.x)

        self.hPrev = hPrev
        self.xPrev = xPrev

        self.xConcatH = np.hstack((x, hPrev))
        self.f = self.sigmoidActivation(np.dot(self.params[0], self.xConcatH) + self.params[4])
        self.i = self.sigmoidActivation(np.dot(self.params[1], self.xConcatH) + self.params[5])
        self.o = self.sigmoidActivation(np.dot(self.params[2], self.xConcatH) + self.params[6])
        self.C = self.tanhActivation(np.dot(self.params[3], self.xConcatH) + self.params[7])
        self.x = (self.C * self.i) + (self.f * xPrev)
        self.h = self.x * self.o

    def carousel(self, dHTop, dXTop, dataLen):
        dX = (self.o * dHTop) + dXTop

        dF = self.dSigmoid(self.f) * (self.xPrev * dX)
        dI = self.dSigmoid(self.i) * (self.C * dX)
        dO = self.dSigmoid(self.o) * (self.x * dHTop)
        dC = self.dTanh(self.C) * (self.i * dX)

        self.params[8] += np.outer(dF, self.xConcatH)
        self.params[9] += np.outer(dI, self.xConcatH)
        self.params[10] += np.outer(dO, self.xConcatH)
        self.params[11] += np.outer(dC, self.xConcatH)
        self.params[12] += dF
        self.params[13] += dI
        self.params[14] += dO
        self.params[15] += dC

        self.dXBottom = dX * self.f
        self.dHBottom = (np.zeros_like(self.xConcatH) +
                         np.dot(self.params[0].T, dF) +
                         np.dot(self.params[1].T, dI) +
                         np.dot(self.params[2].T, dO) +
                         np.dot(self.params[3].T, dC))[dataLen:]

    @staticmethod
    def tanhActivation(val):
        return (np.exp(val) - np.exp(-val)) / (np.exp(val) + np.exp(-val))

    @staticmethod
    def dTanh(vals):
        return 1 - (vals ** 2)

    @staticmethod
    def sigmoidActivation(val):
        return 1 / (1 + np.exp(-val))

    @staticmethod
    def dSigmoid(vals):
        return vals * (1 - vals)
