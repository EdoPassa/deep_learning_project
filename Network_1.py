import random

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Network(object):

    def __init__(self, sizes):
        # The waights and biases are Numpy matrixes initialized at
        # random(Gaussian distributions with mean 0 and standard deviation 1)
        # for example net.weights[1] is storing the weights connecting the second and third layers of neurons
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def feed_forward(self, a): # 'a' is assumed to be an (n, 1) Numpy ndarray
        # Returns the output of the network if 'a' is the input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


    def sdg(self, training_data, epochs, mini_batch_size,
            eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {}/{}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))
