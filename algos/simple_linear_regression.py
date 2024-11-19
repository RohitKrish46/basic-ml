# import numpy as np

# class SimpleLinearRegression:
#     def __init__(self, learning_rate):
#         self.m = 0
#         self.b = 0
#         self.learning_rate = learning_rate

#     def cost_function(self, x, y):
#         total_error = 0
#         for i in range(0, len(x)):
#             total_error += (y[i] - (self.m*x[i] + self.b))**2
#         return total_error/float(len(x))
    
#     def fit(self, x, y, num_iterations):
#         N = float(len(x))
#         for j in range(num_iterations):
#             b_grad = 0
#             m_grad = 0
#             for i in range(len(x)):
#                 b_grad += -(2/N) * (y[i] - ((self.m * x[i]) + self.b))
#                 m_grad += -(2/N) * x[i] * (y[i] - ((self.m * x[i]) + self.b))
#             self.b -= (self.learning_rate * b_grad)
#             self.m -= (self.learning_rate * m_grad)
#         return self
    
# if __name__ == '__main__':
#     x = np.linspace(0, 100, 50)
#     delta = np.random.uniform(-10, 10, x.size)
#     y = 0.5* x + 3 + delta

#     model = SimpleLinearRegression(0.0001)
#     model.fit(x, y, 200)
#     print('Error:', model.cost_function(x, y))




import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class ModelParameters:
    """Stores the learned parameters of the Simple Linear Regression model."""
    slope: float
    intercept: float

class SimpleLinearRegression:
    """
    Simple Linear Regression implementation using gradient descent.
    
    This implementation fits a line y = mx + b to the data by minimizing
    the Mean Squared Error cost function using batch gradient descent.
    
    Attributes:
        learning_rate (float): Step size for gradient descent
        params (ModelParameters): Learned model parameters after fitting
    """
    
    def __init__(self, learning_rate: float):
        """
        Initialize the model with given learning rate.
        
        Args:
            learning_rate (float): Step size for gradient descent
            
        Raises:
            ValueError: If learning rate is not positive
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
            
        self.learning_rate = learning_rate
        self.params = ModelParameters(slope=0.0, intercept=0.0)
        
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate input arrays.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Raises:
            ValueError: If inputs have incorrect shapes or types
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
            
        if X.ndim != 1 or y.ndim != 1:
            raise ValueError("X and y must be 1D arrays")
            
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")
            
        if len(X) < 2:
            raise ValueError("Need at least 2 points to fit a line")
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost function.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            
        Returns:
            float: Mean Squared Error
        """
        self._validate_input(X, y)
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def fit(self, 
           X: np.ndarray, 
           y: np.ndarray, 
           num_iterations: int, 
           verbose: bool = False) -> 'SimpleLinearRegression':
        """
        Fit the linear regression model using gradient descent.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            num_iterations (int): Number of gradient descent iterations
            verbose (bool): If True, print cost during training
            
        Returns:
            self: Returns an instance of self
            
        Raises:
            ValueError: If num_iterations is not positive
        """
        if num_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
            
        self._validate_input(X, y)
        
        N = float(len(X))
        
        for iteration in range(num_iterations):
            # Compute predictions
            y_pred = self.predict(X)
            
            # Compute gradients
            error = y_pred - y
            intercept_gradient = (2/N) * np.sum(error)
            slope_gradient = (2/N) * np.sum(X * error)
            
            # Update parameters using gradient descent
            self.params.intercept -= self.learning_rate * intercept_gradient
            self.params.slope -= self.learning_rate * slope_gradient
            
            # Print progress if verbose
            if verbose and (iteration + 1) % 50 == 0:
                cost = self.compute_cost(X, y)
                print(f"Iteration {iteration + 1}/{num_iterations}, Cost: {cost:.6f}")
                
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
            
        return self.params.slope * X + self.params.intercept
    
    def get_params(self) -> Tuple[float, float]:
        """
        Get the learned model parameters.
        
        Returns:
            Tuple[float, float]: (slope, intercept)
        """
        return self.params.slope, self.params.intercept


def generate_sample_data(
    n_samples: int = 50,
    true_slope: float = 0.5,
    true_intercept: float = 3.0,
    noise_range: Tuple[float, float] = (-10, 10),
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for linear regression.
    
    Args:
        n_samples (int): Number of samples to generate
        true_slope (float): True slope of the line
        true_intercept (float): True intercept of the line
        noise_range (Tuple[float, float]): Range for noise distribution
        random_state (Optional[int]): Random seed for reproducibility
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X and y arrays
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    X = np.linspace(0, 100, n_samples)
    noise = np.random.uniform(noise_range[0], noise_range[1], X.size)
    y = true_slope * X + true_intercept + noise
    
    return X, y


if __name__ == '__main__':
    # Generate sample data
    X, y = generate_sample_data(
        n_samples=50,
        true_slope=0.5,
        true_intercept=3.0,
        random_state=42
    )
    
    # Create and train model
    model = SimpleLinearRegression(learning_rate=0.0001)
    model.fit(X, y, num_iterations=200, verbose=True)
    
    # Print results
    final_cost = model.compute_cost(X, y)
    slope, intercept = model.get_params()
    print("\nFinal Results:")
    print(f"Cost: {final_cost:.6f}")
    print(f"Learned Parameters - Slope: {slope:.4f}, Intercept: {intercept:.4f}")