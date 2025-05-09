import numpy as np
from sklearn.cluster import DBSCAN
import os

class ImageClusterProcessor:
    def __init__(self, comparator, eps=0.2, min_samples=2):
        self.comparator = comparator
        self.eps = eps
        self.min_samples = min_samples

    def build_similarity_matrix(self, image_paths):
        n = len(image_paths)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.comparator.compare_images(image_paths[i], image_paths[j])
                similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
        return similarity_matrix

    def create_clusters(self, image_paths):
        similarity_matrix = self.build_similarity_matrix(image_paths)
        distance_matrix = 1 - similarity_matrix
        clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed').fit_predict(distance_matrix)
        clustered_images = {i: [] for i in set(clusters) if i != -1}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id != -1:
                clustered_images[cluster_id].append(image_paths[idx])
        return clustered_images
