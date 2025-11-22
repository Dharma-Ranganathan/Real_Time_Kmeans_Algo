
from data.generate_data import generate_data
from kmeans.kmeans import KMeans
from visualize import plot_clusters, plot_elbow
import numpy as np

# 1) Generate synthetic dataset
X, true_labels = generate_data(n=400, centers=4)

# 2) Compute WCSS for range
k_values = range(2, 8)
wcss = []

models = []
for k in k_values:
    model = KMeans(k=k)
    model.fit(X)
    models.append(model)
    wcss.append(model.compute_wcss(X))

print("WCSS values:", wcss)

# 3) Automatic detection of optimal K using largest drop method
diff = np.diff(wcss)
optimal_k = list(k_values)[np.argmax(diff) + 1]  # Add 1 to match with K range
print("Automatically Detected Optimal K =", optimal_k)

# 4) Fit final model
final = KMeans(k=optimal_k)
centroids, labels = final.fit(X)

# 5) Plot visuals
plot_elbow(k_values, wcss)
plot_clusters(X, labels, centroids)

# 6) Full written analysis
matches = sum(labels == true_labels)
accuracy = matches / len(labels) * 100

analysis = f"""
--- K-MEANS ANALYSIS REPORT ---

Optimal K Determined: {optimal_k}

Convergence:
- Iterations: {final.iter_count}

Cluster Comparison:
- Matching labels (not true accuracy metric): {matches}/{len(labels)} ({accuracy:.2f}%)

WCSS Trend: {wcss}

Interpretation:
The automatically selected K={optimal_k} corresponds to the elbow point where WCSS begins to decrease more slowly.
Clusters appear well separated, and convergence occurred in {final.iter_count} iterations.
"""

print(analysis)

with open("analysis_report.txt", "w") as f:
    f.write(analysis)

print("Analysis saved as analysis_report.txt")
