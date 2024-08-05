import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import namedtuple


class Neuron:
    """
    One Neuron and its forward connections

    Attributes:
        activa (float): activation strength (0.0 to 1.0),
        weights (list): connections strengths to the set of neurons in the next layer in order
    """

    def __init__(self):
        self.activa = random.random()
        self.weights = []
        self._initialize_neuron()

    def _initialize_neuron(self):
        self.activa = random.random()


class NeuronLayer:
    """
    A neuron layer in a neural network

    Attributes:
        bias (float): bias of layer (might need to move to Neuron class?)
        neurons (list): sorted list of Neurons of `len` `size`
        next_layer (NeuronLayer): reference to the next layer in the network
    """

    def __init__(self, size, weight_limits=(-5.0, 5.0), next_layer=None):
        self.neurons = [Neuron() for _ in range(size)]
        self.bias = 1.0
        self.next_layer = next_layer

        # initializes random weights for each connection to the next layer
        if self.next_layer:
            self._initialize_connections(weight_limits[0], weight_limits[1])

        # create 2d matrix containing the weights of this layer's forward connections
        self.weights_matrix = self._create_weights_matrix()

    def _initialize_connections(self, a, b):
        for neuron in self.neurons:
            neuron.weights = [random.uniform(a, b) for _ in range(len(self.next_layer.neurons))]

    def _create_weights_matrix(self, precision=np.float32):
        weights_list = []
        for neuron in self.neurons:
            weights_list.append(neuron.weights)
        return np.array(weights_list, dtype=precision)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self.neurons):
            self._current_index += 1
            return self.neurons[self._current_index].activa, self.neurons[self._current_index - 1].weights
        else:
            raise StopIteration


class NeuralNetwork:

    def __init__(self):

        # self.layers = {'input': [],
        #                'hidden_layer_1': NeuronLayer(16),
        #                'hidden_layer_2': NeuronLayer(16),
        #                'result': NeuronLayer(10)}

        self.layers = []

        self.result = NeuronLayer(10)
        self.hidden_layer_2 = NeuronLayer(16, next_layer=self.result)
        self.hidden_layer_1 = NeuronLayer(16, next_layer=self.hidden_layer_2)
        self.input = []

    def _generate_result(self):
        for i in self.input:
            for neuron in self.hidden_layer_1.neurons:
                # multiply input by activa?
                test = i * neuron.activa

    def _nudge_peeker(self, expected_result):
        """
        Based on result activation weight, determine nudge factor cost

        :param expected_result: Must be a list of size equal to result
            containing a high weight for desired outcome and low for undesired.
            probably 1.0 for desired result, 0.0 for the rest.
        :return: cost_sum
        """

        nudge_map = []

        for index, neuron in enumerate(self.result.neurons):
            cost = (neuron.activa - expected_result[index]) ** 2
            nudge_map.append(cost if expected_result[index] - neuron.activa > 0 else -cost)

        return nudge_map

    def _back_propagation_layer_1(self, expected_result):
        nudge_list = self._nudge_peeker(expected_result)
        change_tracker = []


if __name__ == '__main__':

    # Normalize data to the range [-5, 5]
    norm = mcolors.Normalize(vmin=-5, vmax=5)
    norm_abs = mcolors.Normalize(vmin=0, vmax=5)

    # Create the colormap
    colors = ["red", "white", "green"]
    cmap = mcolors.LinearSegmentedColormap.from_list("red_to_green", colors)

    test = [random.uniform(-5.0, 5.0) for _ in range(416)]


    def normalize(value, old_min, old_max, new_min, new_max):
        return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


    def get_color(index):
        number = test[index]

        colo = cmap(norm(number))
        alpha = norm_abs(abs(number))
        line = normalize(abs(number), 0.0, 5.0, 0.0, 1.1)

        return colo, alpha, line


    #                  V 2
    label = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    model = NeuralNetwork()

    # print(model.hidden_layer_1.weights_matrix)
    # print(model.hidden_layer_2.weights_matrix)
    #
    # print(type(model.hidden_layer_2.weights_matrix[0][0]))

    # print(model.hidden_layer_1)
    # print(model.hidden_layer_2)

    # model._back_propagation_layer_1(label)

    # Representation of a simple neural network

    size_1 = 16
    size_2 = 10

    # Create a figure and axes
    fig, ax = plt.subplots()

    Coord = namedtuple('Coord', ['x', 'y'])

    n1 = Coord(x=[0] * size_1, y=list(range(size_1)))
    n2 = Coord(x=[1] * size_1, y=list(range(size_1)))
    r1 = Coord(x=[2] * size_2, y=list(range(3, 13)))

    count = 0

    # plt.plot([n1.x, n2.x], [n1.y, n2.y], color='grey', alpha=0.3, linewidth=1)
    for x1, y1 in zip(n1.x, n1.y):
        for x2, y2 in zip(n2.x, n2.y):
            color, norm_alpha, line_width = get_color(count)
            count += 1
            ax.plot([x1, x2], [y1, y2], color=color, alpha=norm_alpha, linewidth=line_width)

    for x1, y1 in zip(n2.x, n2.y):
        for x2, y2 in zip(r1.x, r1.y):
            color, norm_alpha, line_width = get_color(count)
            count += 1
            ax.plot([x1, x2], [y1, y2], color=color, alpha=norm_alpha, linewidth=line_width)

    ax.scatter(n1.x, n1.y, zorder=3, color='C0')
    ax.scatter(n2.x, n2.y, zorder=3, color='C0')
    ax.scatter(r1.x, r1.y, zorder=3, color='C0')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Basic Network')
    plt.show()
