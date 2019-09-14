import numpy as np

POW = 3


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


def print_full_array():
    np.set_printoptions(threshold=np.inf)


class ClusteringAlgorithm:
    def __init__(self, k: int, data: np.ndarray, block_points=None):
        """
        Method initializes object
        :param k: number of clusters, to which data will be divided
        :param data: nparray with shape (h*w, 3),
        containing samples (with shape (3)), that will be divided to clusters
        """
        self.block_points = block_points
        self.dataset = flatten_array(data)
        self.points_count = self.dataset.shape[0]
        self.k = k
        self.centroids = np.empty((self.k, 3))
        self.calculate_initial_centroids()
        self.samples_per_centroid = np.zeros((self.k, 1))

    def set_block_points_as_centroids(self):
        distances_pow = np.ones(self.dataset.shape[0])
        self.centroids[0:self.block_points.shape[0]] = self.block_points
        for block_point in self.block_points:
            distances = calculate_distances_from_point(self.dataset, block_point)
            distances_pow *= np.float_power(distances, POW)
        return distances_pow

    def set_random_centroid(self, i, distances_pow, probability=None):
        # if probability is None:
        #     probability = np.ones(distances_pow.shape)/distances_pow.shape[0]
        data_indexes = range(self.points_count)
        centroid_id = np.random.choice(data_indexes,  p=probability)
        self.centroids[i] = self.dataset[centroid_id]
        distances = calculate_distances_from_point(self.dataset, self.centroids[i])
        distances_pow *= np.float_power(distances, POW)
        return distances_pow

    def calculate_initial_centroids(self):
        """
        Method calculates k centroids (clusters "origins")
        :return: calculated clusters, nparray with shape(k, 3)
        """
        distances_pow = np.ones(self.dataset.shape[0])
        if self.block_points is not None:
            distances_pow = self.set_block_points_as_centroids()
        else:
            distances_pow = self.set_random_centroid(0, distances_pow)
        probability_for_samples = distances_pow/sum(distances_pow)

        # find rest of centroids
        for i in range(self.block_points.shape[0] if self.block_points is not None else 1, self.k):
            distances_pow = self.set_random_centroid(i, distances_pow, probability=probability_for_samples)
            probability_for_samples = distances_pow/sum(distances_pow)
        return self.centroids

    def calculate_distances_for_data(self):
        distances = np.empty((self.k, self.points_count))
        for i, centroid in enumerate(self.centroids):
            distances[i] = calculate_distances_from_point(self.dataset, centroid)
        return distances

    def calculate_clusters(self):
        distances = self.calculate_distances_for_data()
        cluster_ids = np.argmin(distances, axis=0)
        self.centroids = self.calculate_centroids_for_clusters_samples(cluster_ids)
        return cluster_ids

    def calculate_final_clusters(self):
        cluster_ids = np.empty(self.points_count)
        iteration = 0
        while True:
            iteration += 1
            new_cluster_ids = self.calculate_clusters()
            if np.array_equal(cluster_ids, new_cluster_ids) or iteration >= 5:
                break
            cluster_ids = new_cluster_ids
        return self.centroids, self.samples_per_centroid

    def calculate_centroids_for_clusters_samples(self, cluster_ids):
        """
        Method calculates new centroids for given samples' distances
        from existing centroids and their assignment to them
        :param cluster_ids: nparray with shape (h*w, 1)
        :return:
        """
        new_centroids = np.zeros((self.k, 3))
        samples_per_centroid = np.zeros((self.k, 1))
        for sample_id, cluster_id in enumerate(cluster_ids):
            new_centroids[cluster_id] = np.add(new_centroids[cluster_id], self.dataset[sample_id])
            samples_per_centroid[cluster_id] += 1
        denominator = np.ones(new_centroids.shape)
        denominator *= samples_per_centroid
        self.samples_per_centroid = samples_per_centroid.flatten()

        with np.errstate(invalid='ignore'):
            new_centroids = np.divide(new_centroids, denominator)

        if self.block_points is not None:
            new_centroids[0:self.block_points.shape[0]] = self.block_points

        return new_centroids
