import matplotlib.colors as m_colors
import matplotlib.pyplot as plt
import os
import datetime

# Normalize data to the range [-5, 5]
norm = m_colors.Normalize(vmin=-5, vmax=5)
norm_abs = m_colors.Normalize(vmin=0, vmax=5)

# Create the colormap
colors = ["red", "white", "green"]
cmap = m_colors.LinearSegmentedColormap.from_list("red_to_green", colors)


def normalize(value, old_min, old_max, new_min, new_max):
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def get_color(number):
    color = cmap(norm(number))
    norm_alpha = norm_abs(abs(number))
    line_width = normalize(abs(number), 0.0, 5.0, 0.0, 1.1)

    return color, norm_alpha, line_width


def display_network(neuron_layers):
    """
    Ugly ass method
    """

    # Create the coordinates of the circles that represent neurons
    layer_neurons = []
    max_y = 0
    for index, layer in enumerate(neuron_layers):
        l_1 = len(layer.neurons)
        max_y = l_1 if l_1 > max_y else max_y
        adjuster = (max_y - l_1) // 2 if l_1 < max_y else 0
        layer_neurons.append(([index] * len(layer.neurons), list(range(adjuster, len(layer.neurons) + adjuster))))

        plt.scatter(layer_neurons[index][0], layer_neurons[index][1], zorder=3, color='C0')

    color_list = []

    for i in range(len(neuron_layers) - 1):
        for y in neuron_layers[i].weights_2d:
            for z in y:
                color, norm_alpha, line_width = get_color(z)
                color_list.append((color, norm_alpha, line_width))

    count = 0
    # Draw lines between neurons in consecutive layers
    for i in range(len(layer_neurons) - 1):
        x_list_1, y_list_1 = layer_neurons[i]
        x_list_2, y_list_2 = layer_neurons[i + 1]

        # Connect each neuron in layer i to every neuron in layer i + 1
        for x1, y1 in zip(x_list_1, y_list_1):
            for x2, y2 in zip(x_list_2, y_list_2):
                c, a, l1 = color_list[count]
                plt.plot([x1, x2], [y1, y2], color=c, alpha=a, linewidth=l1)
                count += 1

    # Labeling and displaying the plot
    plt.xlabel("Layer Index")
    plt.ylabel("Neuron Index")
    plt.title("Neural Network Neurons with Connections")

    # Define the folder where you want to save the plot
    folder_path = 'C:/Users/Main/Desktop/test/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a unique filename based on the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'neural_network_{timestamp}.png'

    # Save the plot to the specified folder with the new filename
    full_path = os.path.join(folder_path, filename)
    plt.savefig(full_path)

    # plt.show()
