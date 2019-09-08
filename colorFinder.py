import requests
import numpy as np
from PIL import Image
from io import BytesIO

POW = 3

def print_full_array():
    np.set_printoptions(threshold=np.inf)


def download_image_to_array(url):
    """
    Function downloads image from url and saves it to array
    :param url: html link to the picture
    :return: nparray with shape (h, w, 3), containing RGB values for each pixel of the picture
    """
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image_array = np.array(image)
    return image_array


def show_image_from_array(image_array):
    """
    Function shows image based of the given nparray
    :param image_array: nparray with shape (h, w, 3)
    """
    image = Image.fromarray(image_array.astype('uint8'))
    image.show()


def calculate_distance(point_a, point_b):
    """
    Function calculates distance between two points
    :param point_a: fist point's coordinates
    :param point_b: second point's coordinates
    :return: distance between two points
    """
    return np.linalg.norm(point_a-point_b)


def flatten_array(array):
    """
    Function transforms 3-dimensional nparray  to 2-dimensional nparray
    :param array: nparray with shape (h, w, 3)
    :return: nparray with shape (h*w, 3)
    """
    flat_array = []
    for row in array:
        for sample in row:
            flat_array.append(sample)
    return np.array(flat_array)


def calculate_distances_from_point(data, point):
    """
    Function calculates distances for every sample in data to given point
    :param data: nparray with shape (len, 3)
    :param point: nparray with shape (3)
    :return: nparray with shape (len, 1) containing distance for each sample
    """
    distances = np.empty(data.shape[0])
    for (i, sample) in enumerate(data):
        distances[i] = calculate_distance(sample, point)
    return distances


class KMeansClustering:
    def __init__(self, k: int, data: np.ndarray, block_black=False, block_black_and_white=False):
        """
        Method initializes object
        :param k: number of clusters, to which data will be divided
        :param data: nparray with shape (h, w, 3),
        containing samples (with shape (3)), that will be divided to clusters
        """
        self.data = np.array(data)
        self.block_black = block_black
        self.block_black_and_white = block_black_and_white
        self.data_flat = flatten_array(self.data)
        self.samples_count = self.data_flat.shape[0]
        self.data_indexes = [(x, y) for x in range(0, self.data.shape[0]) for y in range(0, self.data.shape[1])]
        self.k = k
        self.centroids = np.empty((self.k, 3))
        self.cluster_map = np.empty((self.data.shape[0], self.data.shape[1]))
        self.cluster_image = np.empty(self.data.shape)
        self.samples_per_centroid = np.zeros((self.k, 1))
        # self.randoms, self.odchylenie = self.initial_centroids()
        # self.clusters = []
        # for i in range(0, self.k):  # create list of lists
        #    self.clusters.append([])

    def calculate_initial_centroids(self):
        """
        Method calculates k centroids (clusters "origins")
        :return: calculated clusters, nparray with shape(k, 3)
        """
        # find first centroid
        data_flat_indexes = range(self.samples_count)

        if self.block_black or self.block_black_and_white:
            self.centroids[0] = np.zeros(3)
        else:
            centroid_id = np.random.choice(data_flat_indexes)
            self.centroids[0] = self.data_flat[centroid_id]

        # calculate probabilities for another centroids
        distances = calculate_distances_from_point(self.data_flat, self.centroids[0])
        distances_square = np.float_power(distances, POW)

        if self.block_black_and_white:
            self.centroids[1] = np.ones(3) * 255
            distances_from_white = calculate_distances_from_point(self.data_flat, self.centroids[1])
            distances_square *= np.float_power(distances_from_white, POW)

        probability_for_samples = distances_square/sum(distances_square)

        # find another centroids
        for i in range(2 if self.block_black_and_white else 1  , self.k):
            centroid_id = np.random.choice(data_flat_indexes, p=probability_for_samples)
            self.centroids[i] = self.data_flat[centroid_id]
            distances *= calculate_distances_from_point(self.data_flat, self.centroids[i])
            distances_square *= np.float_power(distances, POW)
            probability_for_samples = distances_square / sum(distances_square)

        return self.centroids

    def calculate_distances_for_data(self):
        distances = np.empty((self.k, self.samples_count))
        for i, centroid in enumerate(self.centroids):
            distances[i] = calculate_distances_from_point(self.data_flat, centroid)
        return distances

    def calculate_clusters(self):
        distances = self.calculate_distances_for_data()
        cluster_ids = np.argmin(distances, axis=0)
        self.create_cluster_map(cluster_ids)
        self.centroids = self.calculate_centroids_for_clusters_samples(self.data_flat, cluster_ids)
        return cluster_ids

    def create_cluster_map(self, cluster_ids):
        for i in range(self.data.shape[0]):
            self.cluster_map[i] = cluster_ids[i*self.data.shape[1]:(i+1)*self.data.shape[1]]
        print_full_array()

    def create_cluster_image(self):
        for x, y in self.data_indexes:
            centroid_id = int(self.cluster_map[x][y])
            self.cluster_image[x][y] = self.centroids[centroid_id]
        show_image_from_array(self.cluster_image)

    def calculate_colors(self):
        cluster_ids = np.empty(self.samples_count)
        xd = 0
        while True:
            xd += 1
            new_cluster_ids = self.calculate_clusters()
            self.show_palette()
            if np.array_equal(cluster_ids, new_cluster_ids) or xd >= 5:
                break
            cluster_ids = new_cluster_ids

        self.create_cluster_image()

        return self.centroids

    def show_palette(self):
        img = np.ones((int(self.samples_count / 10), 100, 3))
        print(f"samples per centroid {self.samples_per_centroid}")
        h = 0
        for i in range(0, self.k):
            new_h = h + int(self.samples_per_centroid[i] / 10)
            img[h:new_h] *= self.centroids[i]
            print(f"h {h} new_h {new_h} shape {img[h:new_h].shape}")
            h = new_h
        show_image_from_array(img)

    def calculate_centroids_for_clusters_samples(self, data, cluster_ids):
        """
        Method calculates new centroids for given samples' distances
        from existing centroids and their assignment to them
        :param data: nparray with shape (h*w, 3)
        :param cluster_ids: nparray with shape (h*w, 1)
        :return:
        """
        new_centroids = np.zeros((self.k, 3))
        self.samples_per_centroid = np.zeros((self.k, 1))
        for sample_id, cluster_id in enumerate(cluster_ids):
            print(f"sample id {sample_id} cluster id {cluster_id}")
            new_centroids[cluster_id] = np.add(new_centroids[cluster_id], data[sample_id])
            self.samples_per_centroid[cluster_id] += 1
        denominator = np.ones(new_centroids.shape)
        denominator *= self.samples_per_centroid
        new_centroids = np.divide(new_centroids, denominator)

        if self.block_black or self.block_black_and_white:
            new_centroids[0] = self.centroids[0]
        if self.block_black_and_white:
            new_centroids[1] = self.centroids[1]

        return new_centroids


class ColorFinder:
    def __init__(self, url):
        self.image = download_image_to_array(url)
        self.clustering_algorithm = KMeansClustering(k=5, data=self.image, block_black_and_white=True)
        self.clustering_algorithm.calculate_initial_centroids()


link = "https://i.scdn.co/image/0b2f8686d6ff9e978c3c58f93c996aaf94b51fb9"
kk = 5
image_link = download_image_to_array(link)
clustering = KMeansClustering(k=kk, data=image_link, block_black_and_white=True)
show_image_from_array(image_link)
clustering.calculate_initial_centroids()
print(clustering.centroids)
colors = clustering.calculate_colors()
print(f"kolorki:\n{colors}")
# show_image_from_array(img)
# show_image_from_array(clustering.samples_distances_for_centroids())
