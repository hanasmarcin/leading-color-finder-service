import numpy as np
from math import sqrt
from PIL import Image
from tensorflow.keras import models, layers, initializers, Model


def create_model(beta, layer_sizes):
    initializer = initializers.RandomNormal(mean=0., stddev=beta/np.sum(layer_sizes))
    model = models.Sequential()
    model.add(layers.Input(shape=(4,)))
    for i in range(layer_sizes.shape[0]):

        model.add(layers.Dense(layer_sizes[i], activation='tanh', kernel_initializer=initializer))

    return model


def calc_color(R):
    return ((R - np.min(R))/(np.max(R) - np.min(R))*600).astype('uint8')


def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
                # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))


layer = np.array([50, 20, 1])
model = create_model(0.08, layer)
seed = -1


def create_img(x_size, y_size, colors):
    reset_weights(model)

    list = [(x, y, sqrt((x-x_size/2.0)**2 + (y-y_size/2.0)**2), seed) for x in range(x_size) for y in range(y_size)]
    array = np.array(list)
    y = model(array).numpy()
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
