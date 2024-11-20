"""
K-means Clustering Implementation

This module implements the K-means clustering algorithm, an unsupervised learning method that
partitions n observations into k clusters. Each observation belongs to the cluster with the
nearest mean (cluster center or centroid).

The algorithm works through the following steps:
1. Initialize k cluster centers randomly
2. Iterate until convergence:
    a. Assign each point to the nearest cluster center
    b. Update cluster centers by computing the mean of all points in each cluster
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        x1: First point
        x2: Second point

    Returns:
        float: Euclidean distance between x1 and x2
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    """
    K-means clustering algorithm implementation.

    Attributes:
        n_clusters: Number of clusters (K)
        max_iters: Maximum number of iterations for optimization
        plot_steps: Whether to plot intermediate steps
        X: Input data matrix
        n_samples: Number of samples in the dataset
        n_features: Number of features in the dataset
        clusters: List of sample indices for each cluster
        centroids: Array of cluster centers
    """

    def __init__(
        self,
        n_clusters: int = 5,
        max_iters: int = 100,
        plot_steps: bool = False
    ):
        """
        Initialize the KMeans instance.

        Args:
            n_clusters: Number of clusters to form
            max_iters: Maximum number of iterations for optimization
            plot_steps: If True, plot intermediate steps
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # Initialize empty clusters and centroids
        self.clusters: List[List[int]] = []
        self.centroids: np.ndarray = np.array([])
        
        # Will be set during fitting
        self.X: Optional[np.ndarray] = None
        self.n_samples: int = 0
        self.n_features: int = 0

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the K-means model to the data and predict cluster labels.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Cluster labels for each data point
        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.clusters = [[] for _ in range(self.n_clusters)]

        # Initialize centroids randomly
        random_indices = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        self.centroids = np.array([self.X[idx] for idx in random_indices])

        # Optimization loop
        for _ in range(self.max_iters):
            # Assign points to clusters
            self.clusters = self._assign_clusters()
            if self.plot_steps:
                self.plot()

            # Update centroids
            old_centroids = self.centroids.copy()
            self.centroids = self._update_centroids()
            if self.plot_steps:
                self.plot()

            # Check convergence
            if self._has_converged(old_centroids):
                break

        return self._get_cluster_labels()

    def _assign_clusters(self) -> List[List[int]]:
        """
        Assign each data point to the nearest centroid.

        Returns:
            List of lists containing sample indices for each cluster
        """
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._find_closest_centroid(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def _find_closest_centroid(self, sample: np.ndarray) -> int:
        """
        Find the index of the closest centroid to a given sample.

        Args:
            sample: Input sample

        Returns:
            Index of the closest centroid
        """
        distances = [euclidean_distance(sample, point) for point in self.centroids]
        return np.argmin(distances)

    def _update_centroids(self) -> np.ndarray:
        """
        Update centroids by computing the mean of all points in each cluster.

        Returns:
            Updated centroids array
        """
        centroids = np.zeros((self.n_clusters, self.n_features))
        for idx, cluster in enumerate(self.clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids

    def _has_converged(self, old_centroids: np.ndarray) -> bool:
        """
        Check if the algorithm has converged.

        Args:
            old_centroids: Centroids from previous iteration

        Returns:
            True if converged, False otherwise
        """
        distances = [
            euclidean_distance(old_centroids[i], self.centroids[i])
            for i in range(self.n_clusters)
        ]
        return sum(distances) == 0

    def _get_cluster_labels(self) -> np.ndarray:
        """
        Get cluster labels for each data point.

        Returns:
            Array of cluster labels
        """
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def plot(self) -> None:
        """
        Plot the current state of the clustering.
        Shows data points colored by cluster and centroids marked with 'x'.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot points for each cluster
        for i, cluster_indices in enumerate(self.clusters):
            if cluster_indices:  # Only plot if cluster is not empty
                points = self.X[cluster_indices].T
                plt.scatter(*points, label=f'Cluster {i}')
        
        # Plot centroids
        centroids_x = self.centroids[:, 0]
        centroids_y = self.centroids[:, 1]
        plt.scatter(
            centroids_x,
            centroids_y,
            marker='x',
            color='black',
            linewidth=2,
            label='Centroids',
            s=200
        )
        
        plt.legend()
        plt.title('K-means Clustering')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X, y = make_blobs(
        centers=4,
        n_samples=500,
        n_features=2,
        shuffle=True,
        random_state=42
    )
    
    # Determine number of clusters from ground truth
    n_clusters = len(np.unique(y))
    
    # Initialize and fit K-means
    kmeans = KMeans(n_clusters=n_clusters, max_iters=150, plot_steps=True)
    cluster_labels = kmeans.fit_predict(X)