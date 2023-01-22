import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # TODO: initialize W matrices and b vectors
        # layer_sizes example: [4, 10, 2]

        # set number of neurons for layers
        n_input_layer  = layer_sizes[0]
        n_hidden_layer = layer_sizes[1]
        n_output_layer = layer_sizes[2]

        # initialize W matrices with normalize random values
        self.W_layer1 = np.random.normal(size=(n_hidden_layer, n_input_layer))
        self.W_layer2 = np.random.normal(size=(n_output_layer, n_hidden_layer))

        # initialize b vectors with zero values
        self.b_layer1 = np.zeros((n_hidden_layer, 1))
        self.b_layer2 = np.zeros((n_output_layer, 1))

    def activation(self, x):
        
        # TODO: implement Sigmoid function

        sigmoid_output = 1 / (1 + np.exp(-1 * x))
        return sigmoid_output

    def forward(self, x):
        
        # TODO: calculate output values (FeedForward)
        # x example: np.array([[0.1], [0.2], [0.3]])

        # set input_layer values
        a_layer0 = x

        # find values of hidden_layer
        Z1 = np.dot(self.W_layer1, a_layer0) + self.b_layer1
        a_layer1 = self.activation(Z1)
            
        # find values of output_layer
        Z2 = np.dot(self.W_layer2, a_layer1) + self.b_layer2
        a_layer2 = self.activation(Z2)
        output = a_layer2

        return output