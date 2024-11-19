import numpy as np
from dataclasses import dataclass

@dataclass
class ModelParameters:
    """Stores the learned parameters of the Naive Bayes model."""
    classes: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    priors: np.ndarray

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier implementation.
    
    This implementation assumes features follow a normal distribution and are independent.
    The model uses the following formula for classification:
    P(y|X) ∝ P(y) * ∏P(xi|y), where P(xi|y) follows Gaussian distribution.
    
    Attributes:
        params (ModelParameters): Learned model parameters after fitting
    """
    
    def __init__(self):
        self.params = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Naive Bayes model to the training data.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
            
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
        
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Initialize parameters
        means = np.zeros((n_classes, n_features))
        variances = np.zeros((n_classes, n_features))
        priors = np.zeros(n_classes)
        
        # Calculate parameters for each class
        for idx, c in enumerate(classes):
            X_c = X[y == c]
            means[idx, :] = X_c.mean(axis=0)
            variances[idx, :] = X_c.var(axis=0) + 1e-9  # Add small constant to prevent division by zero
            priors[idx] = len(X_c) / n_samples
            
        self.params = ModelParameters(classes, means, variances, priors)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X (np.ndarray): Samples to predict, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted class labels
            
        Raises:
            ValueError: If model is not fitted or input has wrong shape
        """
        if self.params is None:
            raise ValueError("Model must be fitted before making predictions")
            
        if X.ndim != 2:
            raise ValueError(f"X should be 2D array, got shape {X.shape}")
            
        if X.shape[1] != self.params.means.shape[1]:
            raise ValueError(
                f"Expected {self.params.means.shape[1]} features, got {X.shape[1]}"
            )
            
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x: np.ndarray) -> float:
        """
        Predict class for a single sample.
        
        Args:
            x (np.ndarray): Single sample of shape (n_features,)
            
        Returns:
            float: Predicted class label
        """
        posteriors = []
        
        for idx, _ in enumerate(self.params.classes):
            # Calculate log posterior probability for each class
            prior = np.log(self.params.priors[idx])
            likelihood = self._calculate_log_likelihood(idx, x)
            posterior = prior + likelihood
            posteriors.append(posterior)
            
        return self.params.classes[np.argmax(posteriors)]
    
    def _calculate_log_likelihood(self, class_idx: int, x: np.ndarray) -> float:
        """
        Calculate log likelihood of the sample belonging to a specific class.
        
        Args:
            class_idx (int): Index of the class
            x (np.ndarray): Sample features
            
        Returns:
            float: Log likelihood
        """
        mean = self.params.means[class_idx]
        var = self.params.variances[class_idx]
        
        # Log of Gaussian PDF: log(1/√(2πσ²)) - (x-μ)²/(2σ²)
        return np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate prediction accuracy."""
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    # Generate synthetic dataset
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        random_state=123
    )
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=123
    )
    
    # Train and evaluate model
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    
    print(f"Naive Bayes classification accuracy: {accuracy(y_test, predictions):.3f}")