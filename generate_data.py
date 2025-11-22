import numpy as np
from sklearn.datasets import make_blobs

def generate_data(n=300, centers=4):
    X, labels = make_blobs(n_samples=n, centers=centers, cluster_std=1.2, random_state=42)
    return X, labels
