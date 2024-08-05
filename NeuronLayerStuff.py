import numpy as np
import random
from display_network_plot import display_network


class NeuronLayer:
    """
    A neuron layer for a neural network

    Attributes:
        bias (float): Bias of layer
        neurons (list): List of activa values
        next_layer (NeuronLayer): Reference to the next layer in the network
        weights_2d (np.array): Matrix of layer's forward connection weights
    """

    def __init__(self, size):
        self.neurons = [random.random() for _ in range(size)]
        self.bias = 1.0
        self.next_layer = None
        self.weights_2d = None

    def chain_layer(self, next_layer, limits=(-5.0, 5.0)):
        """ Creates new random weights when chaining new layer """
        self.next_layer = next_layer
        self.weights_2d = np.random.uniform(
            low=limits[0], high=limits[1],
            size=(len(self.neurons), len(self.next_layer.neurons))).astype(np.float32)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self.neurons):
            self._current_index += 1
            return self.neurons[self._current_index], self.weights_2d[self._current_index]
        else:
            raise StopIteration


class NeuralNetwork:

    def __init__(self, *args):
        self.neuron_layers = []

        if args and all(isinstance(arg, int) for arg in args):
            self._create_layers(*args)

    def add_layers(self, *args: NeuronLayer):
        for layer in args:
            self.neuron_layers.append(layer)

    def _create_layers(self, *args: int):
        layers = []
        for size in args:
            layers.append(NeuronLayer(size))

        for i in range(len(layers) - 1):
            layers[i].chain_layer(layers[i + 1])

        self.add_layers(*layers)

    def display_network(self):
        display_network(self.neuron_layers)


if __name__ == '__main__':
    nn = NeuralNetwork(16, 16, 10)

    nn.display_network()
