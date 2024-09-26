import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os

pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ddpath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
print(f"Rootpath is set at {ddpath}")

# Load the pickle file
file_path = os.path.join(ddpath, 'data/graphsage_output/unsuprvised_graphsage_entity_embeddings.pkl')

with open(file_path, 'rb') as f:
    embeddings = pickle.load(f)

# Check the type and structure of embeddings
print(type(embeddings))

# Embeddings are in dictionary format, so lets convert them to a NumPy array
if isinstance(embeddings, dict):
    keys = list(embeddings.keys())
    embedding_array = np.array(list(embeddings.values()))
elif isinstance(embeddings, np.ndarray):
    embedding_array = embeddings
else:
    raise ValueError("Unexpected data structure in the pickle file")

print(f"Shape of the embedding array: {embedding_array.shape}")

# Grid search for optimal KMeans parameters
param_grid = {
    'n_clusters': [3, 4, 5, 6, 7, 8, 9, 10],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30],
    'max_iter': [300, 600]
}

kmeans = KMeans(random_state=42)

# Perform grid search
grid_search = GridSearchCV(kmeans, param_grid, cv=5, verbose=1)
grid_search.fit(embedding_array)

# Best model after grid search
best_kmeans = grid_search.best_estimator_
print(f"Best parameters from grid search: {grid_search.best_params_}")

# Get the cluster labels from the best model
labels = best_kmeans.labels_

# Going to get the cluster centers for evaluation as well
cluster_centers = best_kmeans.cluster_centers_

# Print the resulting labels and cluster centers
print(f"Cluster labels: {labels}")
print(f"Cluster centers: {cluster_centers}")

# Analyze clusters if embeddings were a dictionary
if isinstance(embeddings, dict):
    # Create a DataFrame to analyze the clusters
    df = pd.DataFrame({
        'Entity': keys,
        'Cluster': labels
    })

    # Display the DataFrame with cluster assignments
    print(df.head())

    # Grouping by cluster to see the number of entities in each cluster
    print(df.groupby('Cluster').size())

    # Save the cluster labels to a CSV file
    df.to_csv('embedding_clusters.csv', index=False)

# Evaluate the clustering performance
silhouette_avg = silhouette_score(embedding_array, labels)
davies_bouldin = davies_bouldin_score(embedding_array, labels)
calinski_harabasz = calinski_harabasz_score(embedding_array, labels)

print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")

# Reduce dimensionality to 2D for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embedding_array)

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.title("KMeans Clustering of Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.savefig(os.path.join(ddpath, 'data/kmean_5_clustering_graphsage_emb.png'), dpi=300)
#plt.show()

# Save the reduced embeddings if needed
np.save('reduced_embeddings.npy', reduced_embeddings)
