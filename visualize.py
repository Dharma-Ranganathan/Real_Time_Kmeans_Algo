import matplotlib.pyplot as plt

def plot_clusters(X, labels, centroids):
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200)
    plt.title("K-Means Clustering Results")
    plt.show()

def plot_elbow(k_range, wcss):
    plt.plot(k_range, wcss, marker='o')
    plt.title("Elbow Method (WCSS vs K)")
    plt.xlabel("K")
    plt.ylabel("WCSS")
    plt.show()
