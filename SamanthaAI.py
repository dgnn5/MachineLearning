import math as m
import random as r
import copy as c
import ast

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
        self.bias = r.uniform(-0.999999999999999, 0.999999999999999)#Bias has the same range as the weights do

    def __str__(self):#This is what it returns when you print the class
        return ("Hidden Layer 1 Weights: {0}\nHidden Layer 1 Bias: {1}".format(self.weights, self.bias))

    def setValue(self, val):#Setting the node value
        self.value = val

    def getValue(self):#Returning the node value
        return self.value

    def getIndex(self):
        return self.index

    def getBias(self):
        return self.bias

    def setBias(self, value):
        self.bias = value

    def setWeights(self, weightList):#Setting the weights to be equal to an input weightList
        self.weights = weightList

    def getWeights(self):#Returning the weights
        return self.weights

    def createWeights(self, inLayer):#Initializing the weights as random numbers [-0.999999999999999, 0.999999999999999]
        self.weights = []
        for node in inLayer:
            self.weights.append(r.uniform(-0.999999999999999, 0.999999999999999))

    def Propogate(self, inLayer):#Updates the node value based on the node values of the previous layer
        pre_sig_node_val = 0.0
        for i in range(len(self.weights)):#MAYBE REMOVE THIS. NOT SURE IF I CAN REFER TO self.weights HERE. MAYBE CALL ON THE METHOD (ALSO ON THE TWO LINES BELOW, AND THE HiddenLayer CLASS)
            pre_sig_node_val += (self.weights[i] * inLayer[i].getValue())#This is the value of the node before it is run through the sigmoid
        self.setValue(sig(pre_sig_node_val + self.getBias()))#applying the sigmoid


class HiddenLayer:#The rest of the Hidden Layers
    def __init__(self, index):
        self.value = 0.5#This value is the node value
        self.index = index#This value is the index of the hiddenLayers list that this layer lies in
        self.bias = r.uniform(-0.999999999999999, 0.999999999999999)#Bias has the same range as the weights do

    def __str__(self):#if you print this class, this is how it prints
        return ("Hidden Layer Weights: {0}\nHidden Layer Bias: {1}".format(self.weights, self.bias))

    def setValue(self, val):#Setting the node value
        self.value = val

    def getValue(self):#Returning the node value
        return self.value

    def getIndex(self):
        return self.index

    def getBias(self):
        return self.bias

    def setBias(self, value):
        self.bias = value

    def setWeights(self, weightList):
        self.weights = weightList

    def getWeights(self):
        return self.weights

    def createWeights(self, hiddenLayers):#Initializing the weights as random numbers [-0.999999999999999, 0.999999999999999]
        self.weights = []
        for node in hiddenLayers[self.index - 1]:#"hiddenLayers[self.index - 1]" is just the previous hidden layer
            self.weights.append(r.uniform(-0.999999999999999, 0.999999999999999))
#################################
######################  ADDED THE HIDDEN LAYERS PARAMETER BELOW
    def Propogate(self, hiddenLayers):#Updates the node value based on the node values of the previous layer
        pre_sig_node_val = 0.0
        for i in range(len(self.getWeights())):
            try:
                pre_sig_node_val += (self.getWeights()[i] * hiddenLayers[self.index - 1][i].getValue())
            except:
                print(self.index)
        self.setValue(sig(pre_sig_node_val + self.getBias()))


class Parent:#One whole brain
        def __init__(self, hiddenLayers, outLayer, index = 0):#This stores the hiddenLayers list of lists(The Hidden Layers themselves) and the outlayers list as values of the class.
            self.hiddenLayers = hiddenLayers
            self.outLayer = outLayer
            self.index = index

        def __str__(self):
            '''hiddenLayers_string = []
            outLayer_string = []
            for i in range(len(self.hiddenLayers)):
                hiddenLayers_string.append([])
                for x in range(len(self.hiddenLayers[i])):
                    hiddenLayers_string[i].append(str(self.hiddenLayers[i][x]))#might not work
            for i in range(len(self.outLayer)):
                outLayer_string.append(str(self.outLayer[i]))
            return ("Hidden Layers: {0}\nOut Layer: {1}".format(hiddenLayers_string, outLayer_string))'''
            
            hiddenLayers_weights = []
            hiddenLayers_biases = []
            for i in range(len(self.hiddenLayers)):#For layer in self.hiddenLayers
                hiddenLayers_weights.append([])
                hiddenLayers_biases.append([])
                for x in range(len(self.hiddenLayers[i])):#For node in layer
                    hiddenLayers_weights[i].append(self.hiddenLayers[i][x].getWeights())
                    hiddenLayers_biases[i].append(self.hiddenLayers[i][x].getBias())
            outLayer_weights = []
            outLayer_biases = []
            for i in range(len(self.outLayer)):#For node in self.outLayer
                outLayer_weights.append(self.outLayer[i].getWeights())
                outLayer_biases.append(self.outLayer[i].getBias())
            
            parent_list = [hiddenLayers_weights, hiddenLayers_biases, outLayer_weights, outLayer_biases]
            return str(parent_list)

        def setHiddenLayers(self, value):#Overwrites the hiddenLayers list of lists
            self.hiddenLayers = value

        def getHiddenLayers(self):#Returns the hiddenLayers list of lists
            return self.hiddenLayers

        def setOutLayer(self, value):#Overwrites the outLayer list
            self.outLayer = value

        def getOutLayer(self):#Returns the outLayer list
            return self.outLayer

        def getIndex(self):
            return self.index

        def createWeightsAll(self, inLayer):
            for i in range(len(self.getHiddenLayers())):
                for x in range(len(self.getHiddenLayers()[i])):
                    if i == 0:
                        self.getHiddenLayers()[i][x].createWeights(inLayer)
                    else:
                        self.getHiddenLayers()[i][x].createWeights(self.getHiddenLayers())
            for i in range(len(self.getOutLayer())):
                self.getOutLayer()[i].createWeights(self.getHiddenLayers())

        def propogateAll(self, inLayer):#This is just turning the function into a method on the class
            for i in range(len(self.getHiddenLayers())):#For each layer in hiddenLayers
                for x in range(len(self.getHiddenLayers()[i])):#For each not in each layer
                    if i == 0:#If its Layer 1
                        self.getHiddenLayers()[i][x].Propogate(inLayer)
                    else:
                        self.getHiddenLayers()[i][x].Propogate(self.getHiddenLayers())
            for i in range(len(self.getOutLayer())):
                self.getOutLayer()[i].Propogate(self.getHiddenLayers())














def auto_create_skeleton(outLayer_depth: int, num_hiddenLayers: int, hiddenLayers_depths = []):
    outLayer = []
    hiddenLayers = []

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

    return hiddenLayers, outLayer

def auto_create_inLayer(inLayer_values):
    inLayer = []
    for i in inLayer_values:
        inLayer.append(Layer0(i))
    return inLayer

def create_weights(inLayer, hiddenLayers, outLayer):
    for i in range(len(hiddenLayers)):
        for x in range(len(hiddenLayers[i])):
            if i == 0:
                hiddenLayers[i][x].createWeights(inLayer)
            else:
                hiddenLayers[i][x].createWeights(hiddenLayers)
    for i in range(len(outLayer)):
        outLayer[i].createWeights(hiddenLayers)

#print_weights has been replaced by the def __str__ in the Parent class
'''
def print_weights(brain):
    print()
    for i in brain.getHiddenLayers():
        for x in i:
            print("Hidden Layer Weightset: {}".format(x.getWeights()))
            print("Hidden Layer Bias: {}".format(x.getBias()))
    for i in brain.getOutLayer():
        print("Out Layer Weightset: {}".format(i.getWeights()))
        print("Out Layer Bias: {}".format(i.getBias()))
    print()
'''

def get_answer(outLayer):#If outLayer node >= 0.5, they guess "True". If outLayer node < 0.5, they guess "False"
    for i in outLayer:
        answer = i.getValue()
    if answer >= 0.5:
        return True
    else:
        return False

def get_score(number_correct: int, total: int):#Percent Score
    decimal_score = number_correct / total
    percent = decimal_score * 100
    return percent

def take_test(filename: str, brain):#Tests the brain on the training data in filename and returns the Percent score on the test
    f = open(filename, "r")
    number_correct = 0
    total_questions = 0
    for line in f:
        inLayer_values = []
        full_line = line.split()
        for i in range(len(full_line)-1):
            full_line[i] = int(full_line[i])
            inLayer_values.append(full_line[i])
        if full_line[-1] == "True":
            full_line[-1] = True
        elif full_line[-1] == "False":
            full_line[-1] = False
        #print(full_line)
        #print(inLayer_values)
        inLayer = auto_create_inLayer(inLayer_values)
        #for i in inLayer:
        #    print(i.getValue())
        #print_weights(brain)
        brain.propogateAll(inLayer)
        #print_weights(brain)
        answer = get_answer(brain.getOutLayer())
        ##print(brain.getOutLayer()[0].getValue(), answer, full_line[-1])
        if answer == full_line[-1]:
            number_correct += 1
        total_questions += 1
        #print(number_correct)
        #print(brain.getOutLayer()[0].getValue())
        #print(brain.getOutLayer()[0].getWeights())
    f.close()
    #####################################percent = get_score(number_correct, 100000)
    percent = get_score(number_correct, total_questions)
    return percent

def create_generation(generation_size: int, inLayer_depth: int = 5, outLayer_depth: int = 1, num_hiddenLayers: int = 2, hiddenLayers_depths = [2, 2]):
    generation = []
    inLayer = []
    for i in range(inLayer_depth):
        inLayer.append(1)
    inLayer = auto_create_inLayer(inLayer)
    for i in range(generation_size):
        #auto_create_skeleton(outLayer_depth: int, num_hiddenLayers: int, hiddenLayers_depths = [])
        hiddenLayers, outLayer = auto_create_skeleton(outLayer_depth, num_hiddenLayers, hiddenLayers_depths)
        ###hiddenLayers, outLayer = auto_create_skeleton(1, 2, [4, 4])
        create_weights(inLayer, hiddenLayers, outLayer)
        brain = Parent(hiddenLayers, outLayer, i)
        generation.append(brain)
    return generation


#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

#Test_all looks good, but went ahead and added the line to sort brain_score_list here aswell
def test_all(filename: str, generation):
    brain_score_list = []#This is a list that looks like this: [(brain1, score1), (brain2, score2), (brain3, score3)]
    for i in generation:
        brain_score_list.append((i, take_test(filename, i)))#Made this a list of tuples, just in case
    brain_score_list.sort(reverse = True, key = sort_second)
    return brain_score_list

#Find_winners looks good, but I went ahead and swapped the order of the parameters on the .sort() method, so that "reverse = ..." comes before "key = ..."
def find_winners(brain_score_list, number_of_winners: int):
    winners = []
    brain_score_list.sort(reverse = True, key = sort_second)
    for i in range(number_of_winners):
        winners.append(brain_score_list[i])
    return winners

#Decided I don't need this one, since it isn't ever used due to the large amount of memory it would require to store all the generations 
'''
def best_so_far(all_generations):
    total_winners = []
    for generation in all_generations:
        winners = find_winners(generation, 1)[0]
        total_winners.append(winners)
    best = find_winners(total_winners, 1)[0]
    print("Best score so far is: {}\nAnd these are the weights and biases:".format(best[1]))
    print(best[0])
'''

#Best_in_generation looks good
def best_in_generation(generation_brain_score_list):
    #find_winners(generation_brain_score_list, 1) = [(best_brain_1, best_score_1)]
    #Therefore best = (best_brain_1, best_score_1)
    best = find_winners(generation_brain_score_list, 1)[0]
    print("Best score in this generation is: {}\nAnd these are the weights and biases:".format(best[1]))
    print(best[0])

#generation_0_test looks good, but I need to ADD MORE PARAMETERS SO I CAN SET THE BRAIN SIZE                      <-----------Need to add more parameters so I can set the brain size
def generation_0_test(generation_size: int, filename: str, number_of_winners: int, inLayer_depth: int = 5, outLayer_depth: int = 1, num_hiddenLayers: int = 2, hiddenLayers_depths = [2, 2]):
    generation_0 = create_generation(generation_size, inLayer_depth, outLayer_depth, num_hiddenLayers, hiddenLayers_depths)
    brain_score_list = test_all(filename, generation_0)
    winners = find_winners(brain_score_list, number_of_winners)
    #print(brain_score_list)
    print("These numbers are for generation 0\n")
    for i in range(len(winners)):
        print(winners[i][1])
    best_in_generation(brain_score_list)
    return winners

#evolution_step looks good
def evolution_step(winners, generation_size: int, num_parents_per_child: int):
    next_generation = []
    for i in range(generation_size):
        parents = []
        if i == 0:#The first child ALWAYS has the best x of the previous generation as parents
            for x in range(num_parents_per_child):#CHANGED THIS AREA
                parents.append(winners[x][0])#CHANGED THIS AREA
        else:
            for x in range(num_parents_per_child):
                parents.append(r.choice(winners)[0])
        next_generation.append(better_create_child(parents))#CHANGED THIS TO BE BETTER_CREATE_CHILD
    return next_generation


def evolution_big_step(parents, filename: str, num_generations: int, generation_size: int, num_parents_per_child: int, num_parents_per_generation: int):
    #all_brain_score_lists = []
    for i in range(num_generations - 1):#Minus 1 because it doesn't count the 0th generation
        next_generation = evolution_step(parents, generation_size, num_parents_per_child)
        brain_score_list = test_all(filename, next_generation) #####################                           <----------The issue of incorrect score / weight printing in the Evolution function may be in the test_all function
        #all_brain_score_lists.append(brain_score_list)
        parents = find_winners(brain_score_list, num_parents_per_generation)
        print("\nThese numbers are for generation {}\n".format(i + 1))
        #print(brain_score_list)
        for i in range(len(parents)):
            print(parents[i][1])
        best_in_generation(brain_score_list)

def Evolution(filename: str, num_generations: int, generation_size: int, num_parents_per_child: int, num_parents_per_generation: int, inLayer_depth: int = 5, outLayer_depth: int = 1, num_hiddenLayers: int = 2, hiddenLayers_depths = [2, 2]):
    parents = generation_0_test(generation_size, filename, num_parents_per_generation, inLayer_depth, outLayer_depth, num_hiddenLayers, hiddenLayers_depths)
    evolution_big_step(parents, filename, num_generations, generation_size, num_parents_per_child, num_parents_per_generation)

def better_create_child(parent_brains):
    #simple_brain = [[[[#, #, #, #, #], [#, #, #, #, #]], [[#, #], [#, #]]], [[#, #], [#, #]], [[#, #]], [#]]
    #simple_brain[0] = hiddenLayers Weights
    #simple_brain[1] = hiddenLayers Biases
    #simple_brain[2] = outLayer Weights
    #simple_brain[3] = outLayer Biases

    #Makes the parent_brains list of brains into a list of simple_brain lists
    parents = []
    for i in parent_brains:
        parents.append(ast.literal_eval(str(i)))

    #instantiates the child_brain
    child_brain = [[], [], [], []]

    #hiddenLayers Weights
    for i in range(len(parents[0][0])):
        #for each layer in hiddenLayers
        child_brain[0].append([])
        for x in range(len(parents[0][0][i])):
            #for each node in layer i of hiddenLayers
            child_brain[0][i].append([])
            for w in range(len(parents[0][0][i][x])):
                #for each weight in node x of layer i
                temp_weight = 0.0
                for p in range(len(parents)):
                    temp_weight += parents[p][0][i][x][w]
                temp_weight /= len(parents)
                child_brain[0][i][x].append(temp_weight)

    #hiddenLayers Biases
    for i in range(len(parents[0][1])):
        #for each layer in hiddenLayers
        child_brain[1].append([])
        for b in range(len(parents[0][1][i])):
            #for each bias in layer i
            temp_bias = 0.0
            for p in range(len(parents)):
                temp_bias += parents[p][1][i][b]
            temp_bias /= len(parents)
            child_brain[1][i].append(temp_bias)

    #outLayer Weights
    for i in range(len(parents[0][2])):
        #for each node in outLayer
        child_brain[2].append([])
        for w in range(len(parents[0][2][i])):
            #for each weight in node i of outLayer
            temp_weight = 0.0
            for p in range(len(parents)):
                temp_weight += parents[p][2][i][w]
            temp_weight /= len(parents)
            child_brain[2][i].append(temp_weight)

    #outLayer Biases
    for b in range(len(parents[0][3])):
        #for bias in outLayer
        temp_bias = 0.0
        for p in range(len(parents)):
            temp_bias += parents[p][3][b]
        temp_bias /= len(parents)
        child_brain[3].append(temp_bias)

    return full_auto_brain(child_brain)

#Replaced with better_create_child
'''
def create_child(parents, index: int = 0, inLayer = auto_create_inLayer([1, 2, 3, 4, 5])):
    child_outLayer_weights = []
    child_hiddenLayers_weights = []

    child_outLayer_biases = []
    child_hiddenLayers_biases = []

    child_outLayer = []

    child_hiddenLayers = []

    #   THIS IS MAKING A LIST OF THE WEIGHTS FOR THE CHILD OUTLAYER
    
    for x in range(len(parents[0].getOutLayer())):#For each node in the outlayer of any brain (It will be the same value regardless of which parent I pick) (MAYBE ADD BACK IN THE [0] after the .getOutLayer())
        #child_outLayer_weights.append([])#Appends a blank list for each node in the outLayer
        tempList = []
        for w in range(len(parents[0].getOutLayer()[x].getWeights())):#For each weight in node x of the outLayer of any brain ^^
            tempList_2 = []
            for i in parents:#For each brain
                tempList_2.append(i.getOutLayer()[x].getWeights()[w])
            new_weight = average(tempList_2)
            tempList.append(new_weight)
        child_outLayer_weights.append(tempList)

    #    THIS IS MAKING A LIST OF BIASES FOR THE CHILD OUTLAYER

    for x in range(len(parents[0].getOutLayer())):
        tempList = []
        for i in parents:
            tempList.append(i.getOutLayer()[x].getBias())
        new_bias = average(tempList)
        child_outLayer_biases.append(new_bias)
    
    #     HIDDEN LAYERS WEIGHTS GRABBER
    for w in range(len(parents[0].getHiddenLayers())):#For each layer in hiddenLayers
        child_hiddenLayers_weights.append([])
        for x in range(len(parents[0].getHiddenLayers()[w])):#For each node in layer [w]
            tempList = []
            for y in range(len(parents[0].getHiddenLayers()[w][x].getWeights())):#For each weight in node [x] in layer [w]
                tempList_2 = []
                for i in parents:#For each brain
                    tempList_2.append(i.getHiddenLayers()[w][x].getWeights()[y])
                new_weight = average(tempList_2)
                tempList.append(new_weight)
            child_hiddenLayers_weights[w].append(tempList)

    #     HIDDEN LAYERS BIAS GRABBER                                   VVVVVVV MIGHT BE WRONG VVVVVVVV (Don't think so)
    for w in range(len(parents[0].getHiddenLayers())):#For each layer in hiddenLayers
        child_hiddenLayers_biases.append([])
        for x in range(len(parents[0].getHiddenLayers()[w])):#For each node in layer w
            tempList = []
            for i in parents:
                tempList.append(i.getHiddenLayers()[w][x].getBias())
            new_bias = average(tempList)
            child_hiddenLayers_biases[w].append(new_bias)


    #     HIDDEN LAYERS WEIGHTS ASSIGNER

    for i in range(len(child_hiddenLayers_weights)):#For layer in child_hiddenLayers_weights
        child_hiddenLayers.append([])
        for x in range(len(child_hiddenLayers_weights[i])):#For node in layer
            if i == 0:#                                                      <<<<<<<<<<<<<<<<<CHANGED THIS TO BE i=0 instead of x = 0
                child_hiddenLayers[i].append(Layer1())
            else:
                child_hiddenLayers[i].append(HiddenLayer(x))

    for i in range(len(child_hiddenLayers)):#For layer in child_hiddenLayers_weights
        for x in range(len(child_hiddenLayers[i])):#For node in layer
            child_hiddenLayers[i][x].setWeights(child_hiddenLayers_weights[i][x])
            child_hiddenLayers[i][x].setBias(child_hiddenLayers_biases[i][x])#                  <<<<<<<<<<<<MIGHT BE WRONG (Don't think so)


    
    #This is unpacking and assiging all of the weights and biases for the child outLayer
    for i in child_outLayer_weights:#For each node that will be in the child outLayer, where i = the weightList for that node
        child_outLayer.append(HiddenLayer(len(child_hiddenLayers)))#The len(child_hiddenLayers) = the index of the outLayer
    for i in range(len(child_outLayer)):#For each node in the outLayer
        child_outLayer[i].setWeights(child_outLayer_weights[i])#Setting the weightlist value of outLayer node i to the weightlist at index i in the child_outLayer_weights list
        child_outLayer[i].setBias(child_outLayer_biases[i])#Setting the bias value of outLayer node i to the bias at the index i in the child_outLayer_biases list


    child = Parent(child_hiddenLayers, child_outLayer, index)
    return child
'''

def sort_second(val):
    #val is in the format (brain1, score1)
    return val[1]

#Just implimented the averaging into the function itself, no need for this function
'''
def average(values):
    running_total = 0.0
    num_values = len(values)
    for i in values:
        running_total += i
    average = running_total / num_values
    return average
'''


#hiddenLayers_weightsets = [Layer = [[Node_weights], [Node_weights]], [[Node_weights], [Node_weights]], ...]
#hiddenLayers_biases = [Layer = [bias, bias], [bias, bias]]
#outLayer_weightsets = [[Node_weights], [Node_weights], ...]
#outLayer_biases = [Bias, Bias, ...]
def create_brain_from_weights(outLayer_depth: int, num_hiddenLayers: int, hiddenLayers_depths = [], hiddenLayers_weightsets = [], hiddenLayers_biases = [], outLayer_weightsets = [], outLayer_biases = []):
    hiddenLayers, outLayer = auto_create_skeleton(outLayer_depth, num_hiddenLayers, hiddenLayers_depths)
    brain = Parent(hiddenLayers, outLayer)
    for i in range(len(brain.getHiddenLayers())):#For each layer in hiddenLayers
        for x in range(len(brain.getHiddenLayers()[i])):#For each node in layer hiddenLayers[i]
            brain.getHiddenLayers()[i][x].setWeights(hiddenLayers_weightsets[i][x])#Set the weights to a list of weights at hiddenLayers_weightsets[i][x]
            brain.getHiddenLayers()[i][x].setBias(hiddenLayers_biases[i][x])#Set the biases to a single number at hiddenLayers_biases[i][x]
    for i in range(len(brain.getOutLayer())):#For each node in the outLayer
        brain.getOutLayer()[i].setWeights(outLayer_weightsets[i])#Set the weights to a list of weights at outLayer_weightsets[i]
        brain.getOutLayer()[i].setBias(outLayer_biases[i])#Set the bias to a single number at outLayer_biases[i]

    return brain

def auto_create_brain_from_weights(outLayer_depth: int, num_hiddenLayers: int, hiddenLayers_depths = [], parent_list = []):
    hiddenLayers_weights = parent_list[0]
    hiddenLayers_biases = parent_list[1]
    outLayer_weights = parent_list[2]
    outLayer_biases = parent_list[3]
    hiddenLayers, outLayer = auto_create_skeleton(outLayer_depth, num_hiddenLayers, hiddenLayers_depths)
    brain = Parent(hiddenLayers, outLayer)
    for i in range(len(brain.getHiddenLayers())):#For each layer in hiddenLayers
        for x in range(len(brain.getHiddenLayers()[i])):#For each node in layer hiddenLayers[i]
            brain.getHiddenLayers()[i][x].setWeights(hiddenLayers_weights[i][x])#Set the weights to a list of weights at hiddenLayers_weightsets[i][x]
            brain.getHiddenLayers()[i][x].setBias(hiddenLayers_biases[i][x])#Set the biases to a single number at hiddenLayers_biases[i][x]
    for i in range(len(brain.getOutLayer())):#For each node in the outLayer
        brain.getOutLayer()[i].setWeights(outLayer_weights[i])#Set the weights to a list of weights at outLayer_weightsets[i]
        brain.getOutLayer()[i].setBias(outLayer_biases[i])#Set the bias to a single number at outLayer_biases[i]

    return brain

def full_auto_brain(parent_list = []):
    hiddenLayers_weights = parent_list[0]
    hiddenLayers_biases = parent_list[1]
    outLayer_weights = parent_list[2]
    outLayer_biases = parent_list[3]
    outLayer_depth = len(outLayer_weights)
    num_hiddenLayers = len(hiddenLayers_weights)
    hiddenLayers_depths = []
    for i in range(len(hiddenLayers_weights)):
        #for each layer in hiddenLayers_weights
        hiddenLayers_depths.append(len(hiddenLayers_weights[i]))
    hiddenLayers, outLayer = auto_create_skeleton(outLayer_depth, num_hiddenLayers, hiddenLayers_depths)
    brain = Parent(hiddenLayers, outLayer)
    for i in range(len(brain.getHiddenLayers())):#For each layer in hiddenLayers
        for x in range(len(brain.getHiddenLayers()[i])):#For each node in layer hiddenLayers[i]
            brain.getHiddenLayers()[i][x].setWeights(hiddenLayers_weights[i][x])#Set the weights to a list of weights at hiddenLayers_weightsets[i][x]
            brain.getHiddenLayers()[i][x].setBias(hiddenLayers_biases[i][x])#Set the biases to a single number at hiddenLayers_biases[i][x]
    for i in range(len(brain.getOutLayer())):#For each node in the outLayer
        brain.getOutLayer()[i].setWeights(outLayer_weights[i])#Set the weights to a list of weights at outLayer_weightsets[i]
        brain.getOutLayer()[i].setBias(outLayer_biases[i])#Set the bias to a single number at outLayer_biases[i]

    return brain

def back_propogation_step(filename: str, original_brain, mutation_factor: int):
    brain = c.deepcopy(original_brain)#This instantiates brain as a copy of original_brain
    mask = [-1, 0, 1]
    for i in range(len(brain.getHiddenLayers())):#For layer in hiddenLayers
        for x in range(len(brain.getHiddenLayers()[i])):#For node in hiddenLayers[i]
            weights = brain.getHiddenLayers()[i][x].getWeights()
            for w in range(len(weights)):#For weight in weights (list of weights in hiddenLayers[i][x].getWeights())
                weights[w] += (r.choice(mask) * mutation_factor)
            bias = brain.getHiddenLayers()[i][x].getBias()
            bias += (r.choice(mask) * mutation_factor)
            
            brain.getHiddenLayers()[i][x].setWeights(weights)
            brain.getHiddenLayers()[i][x].setBias(bias)

    for i in range(len(brain.getOutLayer())):#For node in outLayer
        weights = brain.getOutLayer()[i].getWeights()
        for x in range(len(weights)):#For weight in weights (outLayer[i])
            weights[x] += (r.choice(mask) * mutation_factor)
        bias = brain.getOutLayer()[i].getBias()
        bias += (r.choice(mask) * mutation_factor)

        brain.getOutLayer()[i].setWeights(weights)
        brain.getOutLayer()[i].setBias(bias)

    original_brain_score = take_test(filename, original_brain)
    brain_score = take_test(filename, brain)
    #print("Original Brain Score: {0}\nBrain Score: {1}".format(original_brain_score, brain_score))
    if brain_score >= original_brain_score:
        return brain, brain_score
    else:#elif parent_score < brain_score
        return original_brain, original_brain_score

def back_propogation(filename: str, brain, mutation_factor: int, propogation_depth: int):
    for i in range(propogation_depth):
        brain, brain_score = back_propogation_step(filename, brain, mutation_factor)
        print("These weights are for the score: {}".format(brain_score))
        print()
        print(brain)
        print()
        print()
    print("\nPROPOGATION DEPTH REACHED.")

def BackPropogation(filename: str, generation_size: int, mutation_factor: int, propogation_depth: int):
    brain_list = create_generation(generation_size)
    brain_score_list = []
    best_score = 0
    best_brain = None
    for i in range(len(brain_list)):
        for x in range(propogation_depth):
            brain_list[i], brain_score = back_propogation_step(filename, brain_list[i], mutation_factor)
        print("These weights are for the score: {}".format(brain_score))
        #brain_score_list.append(brain_score)
        #brain_score_list.sort(reverse = True)
        if brain_score > best_score:
            best_score = brain_score
            best_brain = brain_list[i]
        print()
        print(brain_list[i])
        print()
        print()
    #print("DONE! Best score is: {}".format(brain_score_list[0]))
    print("DONE! Best score is: {}".format(best_score))
    print()
    print(best_brain)








#inLayer = auto_create_inLayer([1, 2, 3, 4, 5])
#hiddenLayers, outLayer = auto_create_skeleton(1, 2, [2, 2])
#create_weights(inLayer, hiddenLayers, outLayer)


#print("This is the inLayer: {}".format(inLayer))
#print("These are the hiddenLayers: {}".format(hiddenLayers))
#print("This is the outLayer: {}".format(outLayer))
#print()
#for i in range(len(inLayer)):
#    print("Here is the value of layer 0: node {0}: {1}".format(i, inLayer[i].getValue()))
#for i in range(len(hiddenLayers)):
#    for x in range(len(hiddenLayers[i])):
#        print("Here is the weightset of layer {0}, node {1} (Node value = {3}): {2}\nAnd here is the bias: {4}".format(i + 1,x , hiddenLayers[i][x].getWeights(), hiddenLayers[i][x].getValue(), hiddenLayers[i][x].getBias()))
#for i in range(len(outLayer)):
#    print("Here is the weightset for the outLayer node {0} (Node value = {2}): {1}\nAnd here is the bias: {3}".format(i, outLayer[i].getWeights(), outLayer[i].getValue(), outLayer[i].getBias()))
#propogate(inLayer, hiddenLayers, outLayer)
#print("After Propogating:")
#for i in range(len(inLayer)):
#    print("Here is the value of layer 0: node {0}: {1}".format(i, inLayer[i].getValue()))
#for i in range(len(hiddenLayers)):
#    for x in range(len(hiddenLayers[i])):
#        print("Here is the weightset of layer {0}, node {1} (Node value = {3}): {2}\nAnd here is the bias: {4}".format(i + 1,x , hiddenLayers[i][x].getWeights(), hiddenLayers[i][x].getValue(), hiddenLayers[i][x].getBias()))
#for i in range(len(outLayer)):
#    print("Here is the weightset for the outLayer node {0} (Node value = {2}): {1}\nAnd here is the bias: {3}".format(i, outLayer[i].getWeights(), outLayer[i].getValue(), outLayer[i].getBias()))

#print()
#print()
#Samantha = Parent(hiddenLayers, outLayer)
#for i in Samantha.getHiddenLayers():
#    for x in i:
#        print(x.getWeights())
#for i in Samantha.getOutLayer():
#    print(i.getWeights())
#print(Samantha.getOutLayer())
'''Samantha.propogateAll(inLayer)
print()
print()
for i in Samantha.getHiddenLayers():
    for x in i:
        print(x.getValue())
for i in Samantha.getOutLayer():
    print(i.getValue())

Samantha.propogateAll(auto_create_inLayer([8, 9, 1, 5, 1]))
print()
print()
for i in Samantha.getHiddenLayers():
    for x in i:
        print(x.getValue())
for i in Samantha.getOutLayer():
    print(i.getValue())

'''
'''
#print("Samantha has taken the test and she got {:.2f}%".format(take_test("Training.txt", Samantha)))
#winners = generation_0_test(5, "Training.txt", 5)
parents = create_generation(3)
for i in parents:
    i.createWeightsAll(inLayer)
for i in range(len(parents)):
    print("\nParent {}:".format(i))
    print_weights(parents[i])
#evolution_full(winners, )
child = create_child(parents)
print("\nChild:")
print_weights(child)
'''
#hiddenLayers_weightsets = [Layer = [[Node_weights], [Node_weights]], [[Node_weights], [Node_weights]], ...]
#hiddenLayers_biases = [Layer = [bias, bias], [bias, bias]]
#outLayer_weightsets = [[Node_weights], [Node_weights], ...]
#outLayer_biases = [Bias, Bias, ...]

#possible_output = [[weightset1, weightset2]


#Evolution(filename: str, num_generations: int, generation_size: int, num_parents_per_child: int, num_parents_per_generation: int, inLayer_depth: int = 5, outLayer_depth: int = 1, num_hiddenLayers: int = 2, hiddenLayers_depths = [2, 2])
#Evolution("Training.txt", 100, 1000, 3, 50)
#######################################################################################################################Evolution("Training_2.txt", 10, 60, 3, 40)
#83.88337720186273%? VVV For Chloe dataset
#Brain = create_brain_from_weights(1, 2, [2, 2], [[[-0.3960067603642052, 0.39926865182843646, -0.018844637953427945, -0.12380053351204097, 0.09236322483611681], [-0.3636583559105566, 0.24795176206060462, 0.10093269197847594, 0.41824669844396745, -0.45127399251213735]], [[-0.13961553443122696, 0.2368173056028231], [-0.7513310080929734, -0.3574767675125116]]], [[-0.16977870144101545, -0.11071017065884749], [0.3941661027975409, 0.17967875530160252]], [[0.266225443978661, -0.6561239865316982]], [0.13029802660852285])

#brain2 = full_auto_brain([[[[-0.6596977456739582, 0.47200568569553586, -0.11023926528988237, 0.09134451869979605, -0.5608768980428919], [0.3584490530107514, 0.5267654852817595, 0.47964449568671474, 0.5508374795318259, 0.30217354936329377]], [[0.6638560438972534, 0.468428582483481], [-0.7346021273108899, -0.13107420799999947]]], [[-0.09893014860476372, 0.19153606306994772], [-0.5000477772410346, 0.558404390006788]], [[-0.24637848738692306, -0.8296855801977955]], [0.5965774556471818]])
#print(take_test("Training_2.txt", brain2))

Evolution("Training_2.txt", 100, 1000, 3, 50, 5, 1, 2, [10, 10])
#brain = full_auto_brain([[[[0.0804362302623051, 0.11187001462470955, -0.04590732234143548, 0.14978822494392116, 0.23233478539259064], [0.04740117185018708, -0.007119261664700432, -0.10859131719556223, 0.2387930079185132, 0.1748794149342238], [-0.353881007538261, 0.34730146422384334, -0.01912088454319249, 0.03319217086096792, 0.0036140475157238425], [0.11082298918409336, -0.06550317002271396, 0.11919223394889544, -0.12010027174033722, -0.20790719670349073]], [[0.08359025396455191, -0.006363520820182504, 0.15477105908606262, 0.15141613838202717], [0.09928914961066727, 0.05421328753879145, -0.2512800206309653, 0.2184988993195187], [-0.21743331202342606, -0.06687128917423339, -0.06194626047099248, -0.2731986615798836], [-0.0015053908447921886, 0.20542642365178584, -0.41490119405794246, 0.0032068445920819264]]], [[-0.03167095816654149, 0.1114472115932019, -0.2457829596818499, -0.10666673123924456], [0.2635682110753187, -0.05412086941891719, -0.10795967311753155, 0.0586204761173752]], [[0.015038088692591172, -0.351345157963714, -0.159689476162887, -0.29903825305949383]], [0.3853261003115171]])
#back_propogation("Training_2.txt", brain, 0.0001, 10000)

#                                     [[[[#, #, #, #, #], [#, #, #, #, #]], [[#, #], [#, #]]], [[#, #], [#, #]], [[#, #]], [#]]
####91.238% VVV For Samantha dataset
####Brain = auto_create_brain_from_weights(1, 2, [2, 2], [[[[-0.393657838364754, 0.2511405050048906, 0.3084797002755429, 0.10167271963088394, -0.16994569660917033], [0.3641581402258271, 0.26814560260335846, -0.03913093728538994, -0.2741193320089045, -0.011399735032822805]], [[-0.0714525400718766, 0.018912829647833623], [0.14308008401278383, 0.16390377222641722]]], [[0.011976546966945133, 0.17910230727729548], [-0.04848170458836652, -0.3093407577765346]], [[-0.1269055167564861, -0.46806920535976954]], [0.28226509850295545]])
#89.0261186474995%? VVV For Chloe dataset
#Brain = create_brain_from_weights(1, 2, [2, 2], [[[-0.37321296900893874, 0.3987318350767525, -0.042992442982257184, -0.12069804116045284, 0.14234532719634438], [-0.3298255054430891, 0.20919368003167185, 0.12544874314187293, 0.3798608570978808, -0.42341153004827503]], [[-0.13031089824589304, 0.1173030274130934], [-0.6901596679131786, -0.31484885052193784]]], [[-0.13749898714983888, -0.069505296605625], [0.36716909593713026, 0.2239892329759011]], [[0.1627763034153046, -0.5664482165842317]], [0.15869794057196474])
###90.84834986839441% VVV For Chloe dataset
###Brain = auto_create_brain_from_weights(1, 2, [2, 2], [[[[-0.3715129690089388, 0.4203318350767527, -0.03789244298225718, -0.11639804116045264, 0.130645327196345], [-0.31472550544308986, 0.19359368003167202, 0.12054874314187314, 0.3627608570978809, -0.3969115300482752]], [[-0.11811089824589294, 0.09600302741309329], [-0.6909596679131788, -0.3049488505219375]]], [[-0.15389898714983885, -0.07400529660562508], [0.33686909593713105, 0.22678923297590023]], [[0.1574763034153043, -0.5349482165842322]], [0.14659794057196496]])
#print_weights(Brain)
#percent_score = take_test("Training_2.txt", Brain)
#print(percent_score)
#back_propogation(filename: str, brain, mutation_factor: int, propogation_depth: int)
####back_propogation("Training.txt", Brain, 0.0001, 200)
#back_propogation("Training_2.txt", Brain, 0.0001, 1000000)
#BackPropogation("Training.txt", 10000, 0.01, 1000)
