'''
Yousif Alnajjar - 112 TP - Model
Sources:
https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/
https://towardsdatascience.com/tutorial-on-lstm-a-computational-perspective-f3417442c2cd#0d00
https://github.com/Manik9/LSTMs
https://github.com/keras-team/keras/blob/v2.11.0/keras/layers/rnn/base_rnn.py#L846-L945
https://github.com/keras-team/keras/blob/v2.11.0/keras/layers/rnn/lstm.py#L382-L893
https://classic.d2l.ai/chapter_recurrent-modern/lstm.html
https://dasha.ai/en-us/blog/log-loss-function
'''

import numpy as np
import yfinance as yf
import copy
import random
from LSTMClasses import Node


def initParams(layers, totalLen):
    wf, wi, wo, wC = (np.random.normal(scale=0.01, size=(layers, totalLen)) for _ in range(4))
    bf, bi, bo, bC = (np.random.normal(scale=0.01, size=(layers)) for _ in range(4))
    dWf, dWi, dWo, dWC = (np.zeros_like(wC) for _ in range(4))
    dBf, dBi, dBo, dBC = (np.zeros_like(bC) for _ in range(4))
    return [wf, wi, wo, wC,
            bf, bi, bo, bC,
            dWf, dWi, dWo, dWC,
            dBf, dBi, dBo, dBC]


# lossCalc and dHCalc sourced from https://github.com/Manik9/LSTMs/blob/master/test.py ToyLossLayer class
def lossCalc(pred, target):
    return (pred[0] - target) ** 2

def dHCalc(pred, target):
    dH = 2 * (pred[0] - target)
    dHs = np.zeros_like(pred)
    dHs[0] = dH
    return dHs


def updateLoss(nodes, layers, targets, dataLen):
    loss = 0
    targetLen = len(targets)

    for i in range(targetLen)[::-1]:
        dX = np.zeros(layers) if (i == targetLen - 1) else nodes[i].dXBottom
        dH = dHCalc(nodes[i].h, targets[i])
        if (i != targetLen - 1):
            dH += nodes[i + 1].dHBottom
        nodes[i].carousel(dH, dX, dataLen)
        loss += lossCalc(nodes[i].h, targets[i])

    return loss


def updateParams(params):
    for i in range(8):
        params[i] -= 0.01 * params[i + 8]
        params[i + 8] = np.zeros_like(params[i])

    return params


def updateNodes(inputs, nodes, layers, params):
    nodes += [Node(params, layers)]

    inputSize = len(inputs)
    if inputSize <= 1:
        nodes[0].hLine(inputs[-1], None, None)
    else:
        hPrev = nodes[inputSize - 2].h
        xPrev = nodes[inputSize - 2].x
        nodes[len(inputs) - 1].hLine(inputs[-1], hPrev, xPrev)
        
    return nodes

def test():
    dataLen = 50
    layers = 100
    totalLen = layers + dataLen
    params = initParams(layers, totalLen)
    y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_array = [np.random.random(dataLen) for _ in y_list]
    nodes = []
    inputs = []

    for cur_iter in range(1001):
        if cur_iter % 100 == 0: print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(y_list)):
            inputs += [input_val_array[ind]]
            nodes = updateNodes(inputs, nodes, layers, params)

        if cur_iter % 100 == 0: print("y_pred = [" +
              ", ".join(["% 2.5f" % nodes[ind].h[0] for ind in range(len(y_list))]) +
              "]", end=", ")

        params, loss = updateParams(params), updateLoss(nodes, layers, y_list, dataLen)
        if cur_iter % 100 == 0: print("loss:", "%.3e" % loss)
        inputs = []