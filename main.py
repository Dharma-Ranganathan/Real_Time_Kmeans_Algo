from data.generate_data import generate_data
from kmeans.kmeans import KMeans
from visualize import plot_clusters, plot_elbow
import numpy as np

# Step 1: Generate synthetic dataset
X, true_labels = generate_data()

# Step 2: Run K-Means for K values 2 to 7 and compute WCSS
wcss = []
for k in range(2, 8):
    model = KMeans(k=k)
    _, labels = model.fit(X)
    wcss.append(model.compute_wcss(X))

print("WCSS values for K=2 to K=7:", wcss)

# Step 3: Plot Elbow Method
plot_elbow(range(2, 8), wcss)

# Step 4: Choose optimal K manually (example: 4)
optimal_k = 4
model = KMeans(k=optimal_k)
centroids, final_labels = model.fit(X)

print("Final centroids:", centroids)

# Step 5: Visualize final clusters
plot_clusters(X, final_labels, centroids)

# Step 6: Simple analysis output
print("Comparison with ground truth (first 20 labels):")
print("Predicted:", final_labels[:20])
print("True:", true_labels[:20])
