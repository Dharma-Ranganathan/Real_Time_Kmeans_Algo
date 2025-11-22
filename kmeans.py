import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        np.random.seed(42)
        random_idxs = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [X[self.labels == i].mean(axis=0) for i in range(self.k)]
            )

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self.centroids, self.labels

    def compute_wcss(self, X):
        return sum(
            np.sum((X[self.labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        )
