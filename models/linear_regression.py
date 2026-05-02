import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]
        
        X_T = X_bias.T
        
        X_T_X = np.dot(X_T, X_bias)
        
        epsilon = 1e-6
        np.fill_diagonal(X_T_X, X_T_X.diagonal() + epsilon)
        
        X_T_X_inv = np.linalg.pinv(X_T_X)
        
        X_T_y = np.dot(X_T, y)
        
        self.weights = np.dot(X_T_X_inv, X_T_y)

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        X_test_bias = np.c_[np.ones(n_samples), X_test]
        
        predictions = np.dot(X_test_bias, self.weights)
        
        return predictions
    
