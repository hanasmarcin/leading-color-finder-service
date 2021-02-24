import numpy as np
from math import sqrt
from PIL import Image


def calc_color(R):
    return ((R - np.min(R))/(np.max(R) - np.min(R))*600).astype('uint8')


def run_nn(network_shape, x, param):
    scale = param/np.sum(network_shape)
    neuron_layers = []
    input_count = x.shape[1]
    first_layer = np.random.normal(loc=0.0, scale=scale, size=(input_count + 1, network_shape[0]))
    neuron_layers.append(first_layer)

    # Create rest of the layers with random coefficients
    for layer_count in range(1, network_shape.shape[0]):
        layer = np.random.normal(loc=0.0, scale=scale, size=(network_shape[layer_count - 1] + 1, network_shape[layer_count]))
        neuron_layers.append(layer)

    for layer in neuron_layers:
        x = np.append(x, np.ones(shape=(x.shape[0], 1)), axis=1)
        print(x.shape)
        print(layer.shape)
        x = np.tanh(x @ layer)

    return x


def create_img(x_size, y_size, colors):

    list = [(x, y, sqrt((x-x_size/2.0)**2 + (y-y_size/2.0)**2), -1) for x in range(x_size) for y in range(y_size)]
    array = np.array(list)
    y = run_nn(np.array((50, 20, 1)), array, 0.08)
    y = y.reshape((x_size, y_size))
    R = calc_color(y)
    print("xd2")
    R = (R % 100)
    image_data = np.zeros([R.shape[0], R.shape[1], 3])

    colors_data = np.array([color.get("pixel_count") for color in colors])
    colors_data = (colors_data / np.sum(colors_data) * 100).astype('uint8')
    colors_count = colors_data.shape[0]
    min_max_R = np.array([(0 if i == 0 else np.sum(colors_data[0:i]), 100 if i == colors_count - 1 else np.sum(colors_data[0:i+1])) for i in range(colors_count)])
    for i in range(colors_count):
        true_indices = np.where(np.logical_and(R >= min_max_R[i, 0], R < min_max_R[i, 1]))
        image_data[true_indices[0], true_indices[1], :] = np.array(colors[i].get("color"))

    img = Image.fromarray(image_data.astype('uint8'), "RGB")
    return img
