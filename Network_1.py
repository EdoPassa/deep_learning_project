import random

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def cost_derivative(output_activations, y):  
    return output_activations-y


class Network(object):

    def __init__(self, sizes):
        # The weights and biases are Numpy matrixes initialized at
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


    def sgd(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        test_data = list(test_data)
        n_test = len(test_data)
        training_data = list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        # nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar
        # to self.biases and self.weights.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # For each training example x, y in mini_batch, accumulate the changes
        # to the weights and biases in nabla_b and nabla_w
        for x, y in mini_batch:
            # This is where we compute the gradient for each training example.
            # backprop does the backpropagation algorithm (see Section 1.3)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # delta_nabla_b and delta_nabla_w are layer-by-layer lists of numpy 
            # arrays, similar to nabla_b and nabla_w, but the values are
            # layer-by-layer gradients
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Update the weights and biases using gradient descent with the
        # accumulated values in nabla_b and nabla_w (see Section 1.3)
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # def cost_derivative(self, output_activations, y):
    #     """Return the vector of partial derivatives \partial C_x /
    #     \partial a for the output activations.
    #     """
    #     return (output_activations-y)
    
    # def save(self, filename):
    #     np.save(filename, self.weights)
    #     np.save(filename, self.biases)

    # def load(self, filename):

    #     self.weights = np.load(filename)
    #     self.biases = np.load(filename)

    # def predict(self, x):

    #     return np.argmax(self.feed_forward(x))
    
    # def predict_proba(self, x):

    #     return self.feed_forward(x)
    
    # def score(self, X, y):

    #     return sum(int(self.predict(x) == y) for x, y in zip(X, y)) / len(X)
    
    # def get_params(self, deep=True):

    #     return {'sizes': self.sizes}
    
    # def set_params(self, **parameters):

    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self
    
    # def __repr__(self):
            
    #     return "Network(sizes={})".format(self.sizes)
    
    # def __str__(self):

    #     return "Network(sizes={})".format(self.sizes)
    
    # def __getstate__(self):

    #     return {'sizes': self.sizes, 'weights': self.weights, 'biases': self.biases}
    
    # def __setstate__(self, state):

    #     self.sizes = state['sizes']
    #     self.weights = state['weights']
    #     self.biases = state['biases']

    
