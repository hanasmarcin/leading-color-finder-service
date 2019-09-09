from ImageUtil import *
from ClusteringAlgorithm import *


class ColorFinder:
    def __init__(self, url, k):
        self.imageUtil = ImageUtil(url)
        self.imageUtil.show()
        self.k = k
        self.block_colors = np.array(((255, 255, 255), (0, 0, 0)))

    def run(self):
        algorithm = ClusteringAlgorithm(k=self.k, data=self.imageUtil.image_array, block_points=self.block_colors)
        colors, pixel_count_for_colors = algorithm.calculate_final_clusters()
        ImageUtil.show_palette(colors, pixel_count_for_colors)


link = "https://i.scdn.co/image/125736ae7eabfbb4edc14307c5bc3313ee54442d"
cf = ColorFinder(link, 5)
cf.run()
