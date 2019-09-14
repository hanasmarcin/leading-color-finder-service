import requests
import numpy as np
from PIL import Image
from io import BytesIO


def show_image_from_array(image_array):
    """
    Function shows image based of the given nparray
    """
    image = Image.fromarray(image_array.astype('uint8'))
    image.show()


class ImageUtil:
    def __init__(self, url):
        self.url = url
        self.image_array = self.download_image_to_array()
        self.pixel_ids = [(x, y) for x in range(0, self.image_array.shape[0]) for y in
                          range(0, self.image_array.shape[1])]
        self.color_image = None

    def download_image_to_array(self):
        """
        Function downloads image from url and saves it to array
        :return: nparray with shape (h, w, 3), containing RGB values for each pixel of the picture
        """
        response = requests.get(self.url)
        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)
        return image_array

    def show(self):
        show_image_from_array(self.image_array)

    def create_color_map(self, color_ids_for_pixels):
        color_map = np.empty(self.image_array.shape[0:1])
        image_width = self.image_array.shape[1]
        for i in range(self.image_array.shape[0]):
            color_map[i] = color_ids_for_pixels[i*image_width: (i+1)*image_width]
        return color_map

    def create_color_image(self, color_ids_for_pixels, colors_rgb):
        cluster_map = self.create_color_map(color_ids_for_pixels)
        self.color_image = np.empty(self.image_array.shape)
        for x, y in self.pixel_ids:
            centroid_id = int(cluster_map[x][y])
            self.color_image[x][y] = colors_rgb[centroid_id]
        return self.color_image

    def show_color_image(self):
        if self.color_image is not None:
            show_image_from_array(self.color_image)
        else:
            raise UserWarning("Color image has not been created.")

    @staticmethod
    def show_palette(colors, pixel_count_for_colors):
        pixel_count = int(np.sum(pixel_count_for_colors))

        img = np.ones((pixel_count, int(pixel_count/pixel_count_for_colors.shape[0]), 3))
        h = 0
        for i, pixel_count_for_color in enumerate(pixel_count_for_colors):
            new_h = h + int(pixel_count_for_color[0])
            print(h)
            print(new_h)
            img[h:new_h] = colors[i]
            h = new_h
        show_image_from_array(img)
