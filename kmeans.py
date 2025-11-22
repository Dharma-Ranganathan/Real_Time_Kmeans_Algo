
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.iter_count = 0

    def fit(self, X):
        np.random.seed(42)
        random_idxs = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_idxs]

        for i in range(self.max_iters):
            self.iter_count = i + 1

            distances = self.compute_distances(X)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = self.compute_new_centroids(X)
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self.centroids, self.labels

    def compute_distances(self, X):
        return np.linalg.norm(X[:, None] - self.centroids, axis=2)

    def compute_new_centroids(self, X):
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])

    def compute_wcss(self, X):
        return sum(np.sum((X[self.labels == i] - self.centroids[i]) ** 2) for i in range(self.k))
