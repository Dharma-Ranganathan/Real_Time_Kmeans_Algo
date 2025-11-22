
import matplotlib.pyplot as plt

def plot_clusters(X, labels, centroids, save=False):
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200)
    plt.title("Final K-Means Cluster Visualization")
    if save:
        plt.savefig("final_clusters.png")
    else:
        plt.show()

def plot_elbow(k_range, wcss, save=False):
    plt.plot(list(k_range), wcss, marker='o')
    plt.xlabel("K")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    if save:
        plt.savefig("elbow_plot.png")
    else:
        plt.show()
