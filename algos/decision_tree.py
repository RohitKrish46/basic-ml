"""
Decision Tree Classifier Implementation

This module implements a Decision Tree Classifier with entropy-based splitting.
The tree is built recursively by selecting the best feature and threshold at each node that maximizes
information gain.

Key Features:
- Entropy-based splitting criteria
- Support for maximum depth limitation
- Minimum samples split criteria
- Random feature selection at each split
"""

from typing import Tuple, Optional
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def calculate_entropy(y: np.ndarray) -> float:
    """
    Calculate the entropy of a dataset's labels.
    
    Args:
        y: Array of class labels
        
    Returns:
        Entropy value of the label distribution
    """
    hist = np.bincount(y)
    probabilities = hist / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


class Node:
    """
    A class representing a node in the decision tree.
    
    Attributes:
        feature: Index of the feature used for splitting
        threshold: Threshold value for the split
        left: Left child node
        right: Right child node
        value: Predicted class label (only for leaf nodes)
    """
    
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['Node'] = None,
        right: Optional['Node'] = None,
        *,
        value: Optional[int] = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self) -> bool:
        """Check if the node is a leaf node."""
        return self.value is not None


class DecisionTree:
    """
    Decision Tree Classifier using entropy and information gain for splitting.
    
    Attributes:
        min_samples_split: Minimum number of samples required to split a node
        max_depth: Maximum depth of the tree
        n_feats: Number of features to consider for each split
        root: Root node of the tree
    """
    
    def __init__(
        self,
        min_samples_split: int = 2,
        max_depth: int = 100,
        n_feats: Optional[int] = None
    ):
        """
        Initialize the Decision Tree Classifier.
        
        Args:
            min_samples_split: Minimum samples required to split a node
            max_depth: Maximum depth of the tree
            n_feats: Number of features to consider for each split
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build the decision tree from training data.
        
        Args:
            X: Training features
            y: Training labels
        """
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grow the decision tree.
        
        Args:
            X: Feature matrix
            y: Target labels
            depth: Current depth in the tree
            
        Returns:
            Node: A new tree node
        """
        n_samples, n_features = X.shape
        n_unique_labels = len(np.unique(y))
        
        # Check stopping criteria
        if (depth >= self.max_depth or
            n_unique_labels == 1 or
            n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))
        
        # Randomly select features to consider for splitting
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y, feat_idxs)
        
        # Create child nodes
        left_idxs, right_idxs = self._split_data(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feat_idxs: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find the best feature and threshold for splitting.
        
        Args:
            X: Feature matrix
            y: Target labels
            feat_idxs: Feature indices to consider
            
        Returns:
            Tuple of (best_feature, best_threshold)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._calculate_information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _calculate_information_gain(
        self,
        y: np.ndarray,
        X_column: np.ndarray,
        threshold: float
    ) -> float:
        """
        Calculate information gain for a potential split.
        
        Args:
            y: Target labels
            X_column: Feature column to split on
            threshold: Threshold value for splitting
            
        Returns:
            Information gain value
        """
        parent_entropy = calculate_entropy(y)
        left_idxs, right_idxs = self._split_data(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate weighted average entropy of children
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = calculate_entropy(y[left_idxs]), calculate_entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        return parent_entropy - child_entropy
    
    def _split_data(
        self,
        X_column: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data based on a feature threshold.
        
        Args:
            X_column: Feature column to split on
            threshold: Threshold value for splitting
            
        Returns:
            Tuple of (left_indices, right_indices)
        """
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs
    
    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Traverse the decision tree to make a prediction.
        
        Args:
            x: Single sample to predict
            node: Current tree node
            
        Returns:
            Predicted class label
        """
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Find the most common class label in a set of labels.
        
        Args:
            y: Array of class labels
            
        Returns:
            Most common class label
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate prediction accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score between 0 and 1
    """
    return np.sum(y_true == y_pred) / len(y_true)


if __name__ == "__main__":
    # Load and prepare data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    
    # Train and evaluate the model
    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = calculate_accuracy(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")