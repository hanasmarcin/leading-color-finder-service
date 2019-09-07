import random
import requests
import numpy as np
from PIL import Image
from io import BytesIO


def download_image_to_array(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image_array = np.array(image)
    return image_array


def show_image_from_array(image_array):
    image = Image.fromarray(image_array.astype('uint8'))
    image.show()


def distance(sample, centroid):
    result = 0
    for dim in range(0, 2):
        result += (sample[dim] - centroid[dim]) ** 2

    result = np.sqrt(result)
    return result


class KMeansClustering:
    def __init__(self, k, data):
        self.data = np.array(data)
        self.data_indexes = [(x, y) for x in range(0, self.data.shape[0]) for y in range(0, self.data.shape[1])]
        self.k = k
        self.centroids = [0, 0, 0]
        self.randoms, self.odchylenie = self.initial_centroids()
        self.clusters = []
        for i in range(0, self.k):    # create list of lists
            self.clusters.append([])

    def initial_centroids(self):
        while True:
            samples = []
            for row in self.data:    # flatten 4-dimensional nparray to array of tuples
                for sample in row:
                    samples.append(tuple(sample))
            randoms = random.sample(samples, self.k)    # find k random samples
            randoms = np.array(randoms)
            odchylenie = np.std(randoms, axis=1)
            break;
        return randoms, odchylenie

    def find_distances_for_sample(self, sample):
        dist = []
        for cluster_id in range(0, self.k):
            centroid = self.centroids[cluster_id]
            dist.append(distance(sample, centroid))
        return dist

    def find_clusters(self):
        for (x, y) in self.data_indexes:
            sample = self.data[x][y].tolist()
            dist = self.find_distances_for_sample(sample)
            min_dist_id = np.asarray(dist).argmin()
            sample.append(dist[min_dist_id])
            self.clusters[min_dist_id].append(sample)
        #print(np.array(self.clusters)[0])
        return np.array(self.clusters)

    def find_average_for_clusters(self):
        colors = np.empty((self.k, 3))
        variances = np.empty((self.k))
        for cluster_with_id in enumerate(self.clusters):
            #print(f"xdxdxd {cluster_with_id[1]}")
            cluster = np.array(cluster_with_id[1])
            #print(cluster)
            #print(cluster.shape)
            colors[cluster_with_id[0]] = np.mean(cluster, axis=0)[:-1]
            variances[cluster_with_id[0]] = np.var(cluster, axis=0)[3]
        return colors.astype(int), np.sum(variances)

    def find_best_colors(self, iterations):
        colors = np.empty((iterations, self.k, 3))
        variances = np.empty((iterations, 2))
        for i in range(0, iterations):
            self.centroids = self.randoms[i*self.k:(i+1)*self.k]
            self.find_clusters()
            colors[i], variances[i][0] = self.find_average_for_clusters()
            #self.centroids, self.odchylenie = self.initial_centroids()
            variances[i][1] = np.sum(self.odchylenie)


        color_id = np.argmin(np.transpose(variances)[0])
        print(f"Wariancje: {variances}\nmin: {np.min(np.transpose(variances)[0])}\nid: {color_id}")
        np.savetxt("variances.csv", variances, delimiter=",")
        return colors[color_id]




link = "https://i.scdn.co/image/39657963338c8bdfd49f4421e9242c6c2f87e8ff"
kk=3
image_link = download_image_to_array(link)
clustering = KMeansClustering(k=kk, data=image_link)
colors = clustering.find_best_colors(1)
img = np.empty((kk*100, 100, 3))
for i in range(0, kk):
    img[100*i:100*i+99][0:99] = colors[i]

#print(F"IMG: {img}")
show_image_from_array(img)
#show_image_from_array(clustering.samples_distances_for_centroids())
