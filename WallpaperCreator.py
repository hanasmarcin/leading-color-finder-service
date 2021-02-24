import numpy as np
from math import sqrt
from PIL import Image


def calc_color(R):
    return ((R - np.min(R))/(np.max(R) - np.min(R))*600).astype('uint8')


def prepare_neuron_layers(input_count, network_shape, param):
    scale = param / np.sum(network_shape)
    neuron_layers = []
    first_layer = np.random.normal(loc=0.0, scale=scale, size=(input_count + 1, network_shape[0]))
    neuron_layers.append(first_layer)

    # Create rest of the layers with random coefficients
    for layer_count in range(1, network_shape.shape[0]):
        layer = np.random.normal(loc=0.0, scale=scale,
                                 size=(network_shape[layer_count - 1] + 1, network_shape[layer_count]))
        neuron_layers.append(layer)
    return neuron_layers


def run_nn(neuron_layers, x):
    for layer in neuron_layers:
        x = np.append(x, np.ones(shape=(x.shape[0], 1)), axis=1)
        x = np.tanh(x @ layer)

    return x


def create_img(x_size, y_size, batch_size, colors):

    list = [(x, y, sqrt((x-x_size/2.0)**2 + (y-y_size/2.0)**2), -1) for x in range(x_size) for y in range(y_size)]
    array = np.array(list)

    nn = prepare_neuron_layers(4, np.array((50, 20, 1)), 0.08)
    y = np.zeros((array.shape[0], 1))
    pixel_count = 0
    while pixel_count < x_size*y_size:
        batch_end = pixel_count + (batch_size if pixel_count + batch_size < x_size*y_size else (x_size*y_size) % batch_size)
        np.copyto(y[pixel_count:batch_end, :], run_nn(nn, array[pixel_count:batch_end, :]))
        pixel_count = batch_end
        # print(y)

    y = y.reshape((x_size, y_size))
    R = calc_color(y)
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
