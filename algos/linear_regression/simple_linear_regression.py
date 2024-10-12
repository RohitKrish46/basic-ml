import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate):
        self.m = 0
        self.b = 0
        self.learning_rate = learning_rate

    def cost_function(self, x, y):
        total_error = 0
        for i in range(0, len(x)):
            total_error += (y[i] - (self.m*x[i] + self.b))**2
        return total_error/float(len(x))
    
    def fit(self, x, y, num_iterations):
        N = float(len(x))
        for j in range(num_iterations):
            b_grad = 0
            m_grad = 0
            for i in range(len(x)):
                b_grad += -(2/N) * (y[i] - ((self.m * x[i]) + self.b))
                m_grad += -(2/N) * x[i] * (y[i] - ((self.m * x[i]) + self.b))
            self.b -= (self.learning_rate * b_grad)
            self.m -= (self.learning_rate * m_grad)
        return self
    
if __name__ == '__main__':
    x = np.linspace(0, 100, 50)
    delta = np.random.uniform(-10, 10, x.size)
    y = 0.5* x + 3 + delta

    model = SimpleLinearRegression(0.0001)
    model.fit(x, y, 200)
    print('Error:', model.cost_function(x, y))