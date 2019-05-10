# necessary packages:
# numpy
# jupyter
# matplotlib
# scipy

# %%
# imports and stuff
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
%matplotlib inline

# %%
# neural network class definition
class neuralNetwork:
    # init neural network (nn)
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # create weight matrizes; wih input to hidden, who hidden to output;
        # w_i_j with node i connecting to node j in next layer;
        # w11 w21
        # w12 w22 ..
        self.wih = np.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # set learning rate
        self.lr = learningrate

        # activation function (sigmoind function)
        self.activation_function = lambda x: sp.expit(x)

        ...

    # train nn
    def train():
        ...

    # query nn
    def query(self, inputs_list):
        # input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals to hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals to final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# %%
# create instance of nn
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

n.query([1.0, 0.5, -1.5])
