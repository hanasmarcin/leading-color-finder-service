from ImageUtil import *
from ClusteringAlgorithm import *


class ColorFinder:
    def __init__(self, url, k):
        self.imageUtil = ImageUtil(url)
        self.k = k
        self.block_colors = np.array(((255, 255, 255), (0, 0, 0)))

    def run(self):
        algorithm = ClusteringAlgorithm(k=self.k, data=self.imageUtil.image_array, block_points=self.block_colors)
        colors, pixel_count_for_colors = algorithm.calculate_final_clusters()
        ImageUtil.show_palette(colors, pixel_count_for_colors)

    def get_colors(self):
        algorithm = ClusteringAlgorithm(k=self.k, data=self.imageUtil.image_array, block_points=self.block_colors)
        colors, pixel_count_for_colors = algorithm.calculate_final_clusters()
        colors_int = [[int(value) for value in color] for color in colors]
        pixel_count_int = [int(count) for count in pixel_count_for_colors]
        return colors_int, pixel_count_int
