
from PIL import Image, ImageFilter
import numpy as np
from math import sqrt
from tensorflow.keras import models, layers, initializers


def create_model(beta, layer_sizes):
    initializer = initializers.RandomNormal(mean=0., stddev=beta/np.sum(layer_sizes))
    model = models.Sequential()
    model.add(layers.Input(shape=(4,)))
    for i in range(layer_sizes.shape[0]):

        model.add(layers.Dense(layer_sizes[i], activation='tanh', kernel_initializer=initializer))

    return model


def calc_color(R):
    return ((R - np.min(R))/(np.max(R) - np.min(R))*600).astype('uint8')


if __name__ == '__main__':
    x_size = 2140
    y_size = 1080
    layer = np.array([50, 20, 1])
    model = create_model(0.08, layer)
    seed = -1

    list = [(x, y, sqrt((x-x_size/2.0)**2 + (y-y_size/2.0)**2), seed) for x in range(x_size) for y in range(y_size)]
    array = np.array(list)
    y = model(array).numpy()
    y = y.reshape((x_size, y_size))
    R = calc_color(y)
    R = (R % 100)
    image_data = np.zeros([R.shape[0], R.shape[1], 3])
    for x in range(R.shape[0]):
        for y in range(R.shape[1]):
            image_data[x, y, :] = \
                np.array((216, 217, 222)) if 0 <= R[x, y] < 25 else \
                np.array((247, 200, 193)) if 25 <= R[x, y] < 50 else \
                np.array((108, 206, 203)) if 50 <= R[x, y] < 75 else \
                np.array((48, 70, 111))

    img = Image.fromarray(image_data.astype('uint8'), "RGB")
    img.save("wallpaper", format='png')
    img.show()

