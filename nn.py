# written in IPython for Jupyter notebook
#
# necessary packages:
# numpy
# jupyter
# matplotlib
# scipy


# to save or load nn, see final two cells


# %%
# imports and stuff
import numpy as np
import pickle
import scipy.special as sp
import scipy.ndimage as spimg
import matplotlib.pyplot as plt
# %matplotlib inline # not necessary in vscode


# %%
# nn class definition
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

        # activation function (sigmoind function); commented out to pickle save
        # self.activation_function = lambda x: sp.expit(x)  # pylint: disable=E1101

    # train nn
    def train(self, inputs_list, targets_list):
        # input lists to 2d arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals to and from hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # changed to be able to pickle save
        # hidden_outputs = self.activation_function(hidden_inputs)
        hidden_outputs = sp.expit(hidden_inputs)

        # calculate signals to and from output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # changed to be able to pickle save
        # final_outputs = self.activation_function(final_inputs)
        final_outputs = sp.expit(final_inputs)

        # calculate error and hidden nodes error
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # update weights for links between hidden and output layers
        self.who += self.lr * \
            np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                   np.transpose(hidden_outputs))

        # update weights for links between input and hidden layers
        self.wih += self.lr * \
            np.dot((hidden_errors * hidden_outputs *
                    (1.0 - hidden_outputs)), np.transpose(inputs))

    # query nn
    def query(self, inputs_list):
        # input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals to and from hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # changed to be able to pickle save
        # hidden_outputs = self.activation_function(hidden_inputs)
        hidden_outputs = sp.expit(hidden_inputs)

        # calculate signals to and from final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # changed to be able to pickle save
        # final_outputs = self.activation_function(final_inputs)
        final_outputs = sp.expit(final_inputs)

        return final_outputs


# %%
# define specs of nn
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# nn learning rate
learning_rate = 0.01


# %%
# create instance of nn
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# %%
# load mnist training dataset
training_data_file = open('mnist_datasets/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# %%
# train nn
# use training data 5 times
epochs = 10
for e in range(epochs):
    print(f'epoch {e + 1}')
    # cycle through training data
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 256.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        # train with rotated variations (1. anti- 2. clockwise)
        inputs_plusx_img = spimg.interpolation.rotate(
            inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        inputs_minusx_img = spimg.interpolation.rotate(
            inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)


# %%
# load mnist test dataset
test_data_file = open('mnist_datasets/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# %%
# test nn
# score
score = []

# cycle through test data
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 256.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        score.append(1)
    else:
        score.append(0)


# %%
# show hit ratio
score_array = np.asarray(score)
print(f'performance: {score_array.sum() / score_array.size}')


# %%
# save trained nn
# with open('trained_nn.pickle', 'wb') as output:
#     pickle.dump(n, output, pickle.HIGHEST_PROTOCOL)
...


# %%
# load trained nn
# with open('trained_nn.pickle', 'rb') as input:
#     n = pickle.load(input)
...
