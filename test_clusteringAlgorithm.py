import numpy as np
import unittest
from unittest import TestCase
from ClusteringAlgorithm import ClusteringAlgorithm


class TestClusteringAlgorithm(TestCase):
    def setUp(self):
        self.data = np.array((((0, 0, 0), (5, 9, 86), (22, 54, 183)),
                              ((4, 10, 86), (6, 8, 82), (22, 53, 183)),
                              ((0, 0, 0), (200, 82, 199), (5, 9, 84))))

        self.distances_expected = np.array(((0.0, 86.61408661412992, 192.06509313251067,
                                             86.6717947200818, 82.60750571225353, 191.78633945096297,
                                             0.0, 293.81116384507925, 84.62860036654276),
                                            (441.6729559300637, 389.32890979222185, 316.0284797292801,
                                             389.34175219208123, 391.07416176474766, 316.6654385941099,
                                             441.6729559300637, 189.97368238785077, 390.2012301364515),
                                            (293.81116384507925, 236.90293370914594, 180.89776118017602,
                                             237.42156599601478, 238.3296037004216, 181.05524018928588,
                                             293.81116384507925, 0.0, 237.86340618094243),
                                            (86.6717947200818, 1.4142135623730951, 108.02314566795395,
                                             0.0, 4.898979485566356, 107.61970079869205,
                                             86.6717947200818, 237.42156599601478, 2.449489742783178),
                                            (191.78633945096297, 107.86102168995063, 1.0,
                                             107.61970079869205, 111.72287142747452, 0.0,
                                             191.78633945096297, 181.05524018928588, 109.66312051004202)))
        self.centroids = np.array(((0, 0, 0), (255, 255, 255), (200, 82, 199), (4, 10, 86), (22, 53, 183)))
        self.cluster_ids_expected = np.array((0, 3, 4, 3, 3, 4, 0, 2, 3))
        self.final_centroids_expected = np.array(((0, 0, 0), (255, 255, 255), (200, 82, 199), (5, 9, 84.5), (22, 53.5, 183)))
        self.samples_per_centroids_expected = np.array((2, 0, 1, 4, 2))
        k = 5
        block_points = np.array(((0, 0, 0), (255, 255, 255)))
        self.algorithm = ClusteringAlgorithm(k=k, data=self.data, block_points=block_points)
        self.algorithm.centroids = self.centroids

    def test_set_block_points_as_centroids(self):
        self.algorithm.centroids = np.empty(self.algorithm.centroids.shape)
        self.algorithm.set_block_points_as_centroids()
        self.assertTrue(np.isin(self.algorithm.centroids[0:2], (0, 0, 0)).any())
        self.assertTrue(np.isin(self.algorithm.centroids[0:2], (255, 255, 255)).any())

    def test_set_random_centroid(self):
        self.algorithm.centroids = np.empty(self.algorithm.centroids.shape)
        self.algorithm.set_random_centroid(0, np.ones(9))
        centroid = self.algorithm.centroids[0]
        self.assertTrue(np.isin(np.array(((5, 9, 86), (22, 54, 183),
                                          (4, 10, 86), (6, 8, 82), (22, 53, 183),
                                          (200, 82, 199), (5, 9, 84))), centroid).any())

    def test_calculate_initial_centroids(self):
        self.algorithm.centroids = np.empty(self.algorithm.centroids.shape)
        initial_centroids = self.algorithm.calculate_initial_centroids()
        for centroid in initial_centroids[2:]:
            self.assertTrue(np.isin(np.array(((5, 9, 86), (22, 54, 183),
                                              (4, 10, 86), (6, 8, 82), (22, 53, 183),
                                              (200, 82, 199), (5, 9, 84))), centroid).any())
        self.assertTrue(np.isin(initial_centroids[0:2], (0, 0, 0)).any())
        self.assertTrue(np.isin(initial_centroids[0:2], (255, 255, 255)).any())
        self.assertTrue(np.unique(initial_centroids, axis=0).shape == initial_centroids.shape)

    def test_calculate_distances_for_data(self):
        distances = self.algorithm.calculate_distances_for_data()
        self.assertTrue(np.allclose(distances, self.distances_expected))

    def test_calculate_clusters(self):
        cluster_ids = self.algorithm.calculate_clusters()
        self.assertTrue(np.allclose(cluster_ids, self.cluster_ids_expected))

    def test_calculate_final_clusters(self):
        final_centroids, samples_per_centroid = self.algorithm.calculate_final_clusters()
        self.assertTrue(np.allclose(final_centroids, self.final_centroids_expected))
        self.assertTrue(np.allclose(samples_per_centroid, self.samples_per_centroids_expected))

    def test_calculate_centroids_for_clusters_samples(self):
        centroids = self.algorithm.calculate_centroids_for_clusters_samples(self.cluster_ids_expected)
        self.assertTrue(np.allclose(centroids, self.final_centroids_expected))


if __name__ == '__main__':
    unittest.main()
