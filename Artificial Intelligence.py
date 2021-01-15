import math as m
import random as r

def sig(x):
    return (1 / (1 + m.exp(-x)))

class Layer0:#inLayer
    def __init__(self, val):
        self.value = val

    def setValue(self, val):
        self.value = val

    def getValue(self):
        return self.value

class Layer1:#Hidden Layer 1
    def __init__(self):#This value is the node value
        self.value = 0.5
        self.index = 0
        self.bias = 0.01

    def setValue(self, val):#Setting the node value
        self.value = val

    def getValue(self):#Returning the node value
        return self.value

    def getIndex(self):
        return self.index

    def getBias(self):
        return self.bias

    def setWeights(self, weightList):#Setting the weights to be equal to an input weightList
        self.weights = weightList

    def getWeights(self):#Returning the weights
        return self.weights

    def createWeights(self):#Initializing the weights as random numbers [0.0, 1.0)
        self.weights = []
        for node in inLayer:
            self.weights.append(r.random())

    def Propogate(self):#Updates the node value based on the node values of the previous layer
        pre_sig_node_val = 0.0
        for i in range(len(self.weights)):#MAYBE REMOVE THIS. NOT SURE IF I CAN REFER TO self.weights HERE. MAYBE CALL ON THE METHOD (ALSO ON THE TWO LINES BELOW, AND THE HiddenLayer CLASS)
            pre_sig_node_val += (self.weights[i] * inLayer[i].getValue())#This is the value of the node before it is run through the sigmoid
        self.setValue(sig(pre_sig_node_val + self.getBias()))#applying the sigmoid


class HiddenLayer:#The rest of the Hidden Layers
    def __init__(self, index):
        self.value = 0.5#This value is the node value
        self.index = index#This value is the index of the hiddenLayers list that this layer lies in
        self.bias = 0.01#Defining the bias

    def setValue(self, val):#Setting the node value
        self.value = val

    def getValue(self):#Returning the node value
        return self.value

    def getIndex(self):
        return self.index

    def getBias(self):
        return self.bias

    def setWeights(self, weightList):
        self.weights = weightList

    def getWeights(self):
        return self.weights

    def createWeights(self):#Initializing the weights as random numbers [0.0, 1.0)
        self.weights = []
        for node in hiddenLayers[self.index - 1]:#"hiddenLayers[self.index - 1]" is just the previous hidden layer
            self.weights.append(r.random())

    def Propogate(self):#Updates the node value based on the node values of the previous layer
        pre_sig_node_val = 0.0
        for i in range(len(self.weights)):
            pre_sig_node_val += (self.weights[i] * hiddenLayers[self.index - 1][i].getValue())
        self.setValue(sig(pre_sig_node_val + self.getBias()))
        

def create_skeleton(inLayer_depth: int, outLayer_depth: int, num_hiddenLayers: int):
    inLayer = []
    outLayer = []
    hiddenLayers = []
    for i in range(inLayer_depth):
        initial_inLayer_value = int(input('What is the value of Layer 0 (The "Input Layer"), Node {}?\n'.format(i)))
        inLayer.append(Layer0(initial_inLayer_value))

    for i in range(num_hiddenLayers):
        tempLayer = []
        hiddenLayer_depth = int(input("How many nodes will be in Hidden Layer {}?\n".format(i)))
        if i == 0:#If it is Layer 1
            for x in range(hiddenLayer_depth):
                tempLayer.append(Layer1())#I DON'T KNOW IF I DID THIS CORRECT. MAYBE REMOVE THE PARENTHESIS AFTER THE WORD "Layer1"
            '''for w in range(len(tempLayer)):
                tempLayer[w].createWeights()'''
            hiddenLayers.append(tempLayer)
            tempLayer = []
        else:
            for x in range(hiddenLayer_depth):
                tempLayer.append(HiddenLayer(i))
            '''for w in range(len(tempLayer)):
                tempLayer[w].createWeights()'''
            hiddenLayers.append(tempLayer)
            tempLayer = []

    for i in range(outLayer_depth):
        if len(hiddenLayers) > 0:
            outLayer.append(HiddenLayer(len(hiddenLayers)))
            '''outLayer[i].createWeights()'''
        else:
            outLayer.append(Layer1())
            '''outLayer[i].createWeights()'''

    return inLayer, hiddenLayers, outLayer


def auto_create_skeleton(inLayer_values, outLayer_depth: int, num_hiddenLayers: int, hiddenLayers_depths = []):
    inLayer = []
    outLayer = []
    hiddenLayers = []
    for i in inLayer_values:
        inLayer.append(Layer0(i))

    for i in range(num_hiddenLayers):
        tempLayer = []
        if i == 0:
            for x in range(hiddenLayers_depths[i]):
                tempLayer.append(Layer1())
            hiddenLayers.append(tempLayer)
            tempLayer = []
        else:
            for x in range(hiddenLayers_depths[i]):
                tempLayer.append(HiddenLayer(i))
            hiddenLayers.append(tempLayer)
            tempLayer = []

    for i in range(outLayer_depth):
        if len(hiddenLayers) > 0:
            outLayer.append(HiddenLayer(len(hiddenLayers)))
        else:
            outLayer.append(Layer1())

    return inLayer, hiddenLayers, outLayer


def create_weights(inLayer, hiddenLayers, outLayer):
    for i in range(len(hiddenLayers)):
        for x in range(len(hiddenLayers[i])):
            hiddenLayers[i][x].createWeights()
    for i in range(len(outLayer)):
        outLayer[i].createWeights()

def propogate(inLayer, hiddenLayers, outLayer):
    for i in range(len(hiddenLayers)):
        for x in range(len(hiddenLayers[i])):
            hiddenLayers[i][x].Propogate()
    for i in range(len(outLayer)):
        outLayer[i].Propogate()


inLayer_depth = int(input("How many nodes will be in the Input Layer?\n"))
outLayer_depth = int(input("How many nodes will be in the Output Layer?\n"))
num_hiddenLayers = int(input("How many Hidden Layers will there be?\n"))
inLayer, hiddenLayers, outLayer = create_skeleton(inLayer_depth, outLayer_depth, num_hiddenLayers)
create_weights(inLayer, hiddenLayers, outLayer)
#print("This is the inLayer: {}".format(inLayer))
#print("These are the hiddenLayers: {}".format(hiddenLayers))
#print("This is the outLayer: {}".format(outLayer))
#print()
for i in range(len(inLayer)):
    print("Here is the value of layer 0: node {0}: {1}".format(i, inLayer[i].getValue()))
for i in range(len(hiddenLayers)):
    for x in range(len(hiddenLayers[i])):
        print("Here is the weightset of layer {0}, node {1} (Node value = {3}): {2}\nAnd here is the bias: {4}".format(i + 1,x , hiddenLayers[i][x].getWeights(), hiddenLayers[i][x].getValue(), hiddenLayers[i][x].getBias()))
for i in range(len(outLayer)):
    print("Here is the weightset for the outLayer node {0} (Node value = {2}): {1}\nAnd here is the bias: {3}".format(i, outLayer[i].getWeights(), outLayer[i].getValue(), outLayer[i].getBias()))
propogate(inLayer, hiddenLayers, outLayer)
print("After Propogating:")
for i in range(len(inLayer)):
    print("Here is the value of layer 0: node {0}: {1}".format(i, inLayer[i].getValue()))
for i in range(len(hiddenLayers)):
    for x in range(len(hiddenLayers[i])):
        print("Here is the weightset of layer {0}, node {1} (Node value = {3}): {2}\nAnd here is the bias: {4}".format(i + 1,x , hiddenLayers[i][x].getWeights(), hiddenLayers[i][x].getValue(), hiddenLayers[i][x].getBias()))
for i in range(len(outLayer)):
    print("Here is the weightset for the outLayer node {0} (Node value = {2}): {1}\nAnd here is the bias: {3}".format(i, outLayer[i].getWeights(), outLayer[i].getValue(), outLayer[i].getBias()))
