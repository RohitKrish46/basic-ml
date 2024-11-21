import numpy as np
from typing import Tuple, List
from collections import Counter
from decision_tree import DecisionTree


def bootstrap_sample(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a bootstrap sample of the input data.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Input labels
    
    Returns:
        Tuple of sampled features and labels
    """
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]


def most_common_label(y: np.ndarray) -> np.ndarray:
    """
    Find the most common label in an array.
    
    Args:
        y (np.ndarray): Array of labels
    
    Returns:
        The most common label
    """
    counter = Counter(y)
    return counter.most_common(1)[0][0]


class RandomForest:
    """
    Random Forest Classifier implementation.
    
    Attributes:
        n_trees (int): Number of trees in the forest
        min_samples_split (int): Minimum samples required to split a node
        max_depth (int): Maximum depth of the trees
        n_feats (int, optional): Number of features to consider for splitting
        trees (List[DecisionTree]): List of decision trees in the forest
    """
    
    def __init__(
        self, 
        n_trees: int = 100, 
        min_samples_split: int = 2, 
        max_depth: int = 100, 
        n_feats: int = None
    ):
        """
        Initialize the Random Forest.
        
        Args:
            n_trees (int): Number of trees to create
            min_samples_split (int): Minimum samples to split a node
            max_depth (int): Maximum tree depth
            n_feats (int, optional): Number of features to consider
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees: List[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Random Forest to the training data.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
        """
        self.trees = []
        for _ in range(self.n_trees):
            # Create and train individual trees on bootstrap samples
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats
            )
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input features.
        
        Args:
            X (np.ndarray): Input features
        
        Returns:
            np.ndarray: Predicted labels
        """
        # Predict using each tree and take the most common prediction
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        y_pred = np.array([most_common_label(tree_pred) for tree_pred in tree_predictions])
        return y_pred


def main():
    """
    Example usage and testing of the Random Forest implementation.
    """
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the classifier
    clf = RandomForest(n_trees=10, max_depth=10)
    clf.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()