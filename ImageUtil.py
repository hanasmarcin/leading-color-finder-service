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

    # def create_cluster_image(self, colors, colors_for_pixels):
    #     color_map =
    #     for x, y in self.pixel_ids:
    #         centroid_id = int(self.cluster_map[x][y])
    #         self.cluster_image[x][y] = self.centroids[centroid_id]
    #     show_image_from_array(self.cluster_image)

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
