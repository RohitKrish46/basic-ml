import numpy as np
from collections import Counter
from typing import Tuple, Union, Optional
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin

@dataclass
class Distance:
    """Stores distance calculation methods."""
    @staticmethod
    def euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            x1 (np.ndarray): First point
            x2 (np.ndarray): Second point
            
        Returns:
            float: Euclidean distance between x1 and x2
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    @staticmethod
    def manhattan(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate the Manhattan distance between two points.
        
        Args:
            x1 (np.ndarray): First point
            x2 (np.ndarray): Second point
            
        Returns:
            float: Manhattan distance between x1 and x2
        """
        return np.sum(np.abs(x1 - x2))

class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors classifier implementation.
    
    This implementation classifies samples based on majority voting of K nearest neighbors
    using specified distance metric.
    
    Attributes:
        k (int): Number of neighbors to use for classification
        distance_metric (str): Distance metric to use ('euclidean' or 'manhattan')
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
    """
    
    def __init__(self, k: int = 3, distance_metric: str = 'euclidean'):
        """
        Initialize KNN classifier.
        
        Args:
            k (int): Number of neighbors to use
            distance_metric (str): Distance metric to use
            
        Raises:
            ValueError: If k is not positive or distance_metric is not supported
        """
        if k <= 0:
            raise ValueError("k must be positive")
            
        self.k = k
        self.distance_functions = {
            'euclidean': Distance.euclidean,
            'manhattan': Distance.manhattan
        }
        
        if distance_metric not in self.distance_functions:
            raise ValueError(f"Unsupported distance metric. Choose from: {list(self.distance_functions.keys())}")
            
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestNeighbors':
        """
        Fit the KNN classifier (store the training data).
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            
        Returns:
            self: Returns an instance of self
            
        Raises:
            ValueError: If input arrays have incorrect shapes or types
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
            
        if X.ndim != 2:
            raise ValueError(f"X should be 2D array, got shape {X.shape}")
            
        if y.ndim != 1:
            raise ValueError(f"y should be 1D array, got shape {y.shape}")
            
        if len(X) != len(y):
            raise ValueError(f"X and y must have same number of samples. Got {len(X)} and {len(y)}")
            
        if len(X) < self.k:
            raise ValueError(f"Number of training samples ({len(X)}) must be greater than k ({self.k})")
        
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X (np.ndarray): Samples to predict
            
        Returns:
            np.ndarray: Predicted class labels
            
        Raises:
            ValueError: If model is not fitted or input has wrong shape
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before making predictions")
            
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
            
        if X.ndim != 2:
            raise ValueError(f"X should be 2D array, got shape {X.shape}")
            
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Expected {self.X_train.shape[1]} features, got {X.shape[1]}"
            )
            
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x: np.ndarray) -> Union[int, str]:
        """
        Predict class for a single sample.
        
        Args:
            x (np.ndarray): Single sample to predict
            
        Returns:
            Union[int, str]: Predicted class label
        """
        # Calculate distances to all training samples
        distances = [
            self.distance_functions[self.distance_metric](x, x_train)
            for x_train in self.X_train
        ]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Return most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def evaluate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier: KNearestNeighbors,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """
    Evaluate the classifier using train-test split.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        classifier (KNearestNeighbors): Classifier instance
        test_size (float): Proportion of dataset to use for testing
        random_state (Optional[int]): Random seed for reproducibility
        
    Returns:
        Tuple[float, np.ndarray]: (accuracy, predictions)
    """
    from sklearn.model_selection import train_test_split
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Train and evaluate
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = np.mean(y_test == predictions)
    
    return accuracy, predictions


if __name__ == "__main__":
    from sklearn import datasets
    
    # Load iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # Create and evaluate classifier
    classifier = KNearestNeighbors(k=3, distance_metric='euclidean')
    accuracy, predictions = evaluate_classifier(
        X, y,
        classifier=classifier,
        test_size=0.2,
        random_state=42
    )
    
    print(f"KNN Classification Accuracy: {accuracy:.3f}")
    
    # Example of using different distance metric
    classifier_manhattan = KNearestNeighbors(k=3, distance_metric='manhattan')
    accuracy_manhattan, _ = evaluate_classifier(
        X, y,
        classifier=classifier_manhattan,
        test_size=0.2,
        random_state=42
    )
    
    print(f"KNN Classification Accuracy (Manhattan): {accuracy_manhattan:.3f}")