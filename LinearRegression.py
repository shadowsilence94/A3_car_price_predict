import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import mean_squared_error

class LinearRegression:
    """
    Enhanced Linear Regression class with regularization, Xavier initialization, momentum, and feature importance.
    """
    
    def __init__(self, regularization=None, learning_rate=0.01, max_iter=1000, 
                 init_method='xavier', use_momentum=False, momentum=0.9, 
                 batch_type='batch', batch_size=32, **kwargs):
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_method = init_method
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.theta = None
        self.history = {'mse': [], 'r2': []}
        self.prev_step = 0
        self.feature_names = None

    def _initialize_theta(self, n_features):
        """
        Initializes weights (theta) using 'zeros' or 'xavier' method.
        The size of theta must be n_features + 1 to account for the bias.
        """
        if self.init_method == 'xavier':
            # Xavier Initialization for feature weights + 1 for bias term
            limit = np.sqrt(6.0 / (n_features + 1))
            self.theta = np.random.uniform(-limit, limit, size=n_features + 1)
        else: # 'zeros'
            self.theta = np.zeros(n_features + 1)

    def fit(self, X, y, feature_names=None):
        """
        Fits the model to the training data.
        """
        if feature_names is not None:
            self.feature_names = feature_names
        
        m, n = X.shape
        self._initialize_theta(n)
        
        X_b = np.c_[np.ones((m, 1)), X]
        
        if self.batch_type == 'batch':
            self._fit_batch_gd(X_b, y)
        elif self.batch_type == 'mini-batch':
            self._fit_mini_batch_gd(X_b, y)
        else: # stochastic
            self._fit_stochastic_gd(X_b, y)
    
    def _fit_batch_gd(self, X_b, y):
        for i in range(self.max_iter):
            y_pred = X_b.dot(self.theta)
            residuals = y_pred - y
            gradient = X_b.T.dot(residuals) / len(y)
            
            if self.regularization:
                reg_gradient = self.regularization.derivation(self.theta)
                gradient[1:] += reg_gradient[1:]
            
            if self.use_momentum:
                step = self.learning_rate * gradient
                self.theta -= step + self.momentum * self.prev_step
                self.prev_step = step
            else:
                self.theta -= self.learning_rate * gradient

            if np.isnan(self.theta).any():
                print(f"Warning: NaN detected in batch gradient descent. Stopping early at iteration {i}.")
                self.theta = np.full(self.theta.shape, np.nan)
                break
            
            self._record_metrics(X_b[:, 1:], y, y_pred)
            
    def _fit_mini_batch_gd(self, X_b, y):
        m = len(y)
        for i in range(self.max_iter):
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X_b[indices], y[indices]
            
            for j in range(0, m, self.batch_size):
                X_batch = X_shuffled[j:j + self.batch_size]
                y_batch = y_shuffled[j:j + self.batch_size]
                
                y_pred = X_batch.dot(self.theta)
                residuals = y_pred - y_batch
                gradient = X_batch.T.dot(residuals) / len(y_batch)
                
                if self.regularization:
                    reg_gradient = self.regularization.derivation(self.theta)
                    gradient[1:] += reg_gradient[1:]

                if self.use_momentum:
                    step = self.learning_rate * gradient
                    self.theta -= step + self.momentum * self.prev_step
                    self.prev_step = step
                else:
                    self.theta -= self.learning_rate * gradient
            
            if np.isnan(self.theta).any():
                print(f"Warning: NaN detected in mini-batch gradient descent. Stopping early at iteration {i}.")
                self.theta = np.full(self.theta.shape, np.nan)
                break

            self._record_metrics(X_b[:, 1:], y, X_b.dot(self.theta))

    def _fit_stochastic_gd(self, X_b, y):
        m = len(y)
        for i in range(self.max_iter):
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            y_pred = xi.dot(self.theta)
            residuals = y_pred - yi
            gradient = xi.T.dot(residuals)
            
            if self.regularization:
                reg_gradient = self.regularization.derivation(self.theta)
                gradient[1:] += reg_gradient[1:]

            if self.use_momentum:
                step = self.learning_rate * gradient
                self.theta -= step + self.momentum * self.prev_step
                self.prev_step = step
            else:
                self.theta -= self.learning_rate * gradient
            
            if np.isnan(self.theta).any():
                print(f"Warning: NaN detected in stochastic gradient descent. Stopping early at iteration {i}.")
                self.theta = np.full(self.theta.shape, np.nan)
                break

            if i % 100 == 0:
                self._record_metrics(X_b[:, 1:], y, X_b.dot(self.theta))
    
    def _record_metrics(self, X, y, y_pred):
        """
        Records MSE and R2 score for tracking convergence.
        """
        mse = np.mean((y_pred - y) ** 2)
        r2 = self.r2(X, y)
        self.history['mse'].append(mse)
        self.history['r2'].append(r2)

    def predict(self, X):
        """
        Predicts the output for new data.
        """
        if np.isnan(self.theta).any():
            return np.full(X.shape[0], np.nan)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

    def r2(self, X, y):
        """
        Calculates the R-squared (R²) score.
        """
        y_pred = self.predict(X)
        if np.isnan(y_pred).any():
            return np.nan
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0
        return 1 - (ss_res / ss_tot)
        
    def plot_feature_importance(self):
        """
        Plots feature importance based on absolute coefficient values.
        """
        if self.theta is None or np.isnan(self.theta).any():
            print("Model has not been fitted or contains NaN coefficients.")
            return

        importances = np.abs(self.theta[1:])
        
        if self.feature_names is not None:
            if len(importances) == len(self.feature_names):
                features = self.feature_names
            else:
                features = [f'F{i+1}' for i in range(len(importances))]
        else:
            features = [f'F{i+1}' for i in range(len(importances))]

        sorted_indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance from Coefficients")
        plt.bar(range(len(features)), importances[sorted_indices], align='center')
        plt.xticks(range(len(features)), [features[i] for i in sorted_indices], rotation=90)
        plt.tight_layout()
        plt.show()

class Ridge(LinearRegression):
    """Ridge Regression with L2 regularization."""
    
    def __init__(self, lambda_reg=0.1, **kwargs):
        regularization = RidgePenalty(lambda_reg)
        super().__init__(regularization=regularization, **kwargs)

class Lasso(LinearRegression):
    """Lasso Regression with L1 regularization."""
    
    def __init__(self, lambda_reg=0.1, **kwargs):
        regularization = LassoPenalty(lambda_reg)
        super().__init__(regularization=regularization, **kwargs)

# Penalty classes
class LassoPenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)

class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(theta ** 2)

    def derivation(self, theta):
        return self.l * 2 * theta

class ElasticPenalty:
    def __init__(self, lambda_reg, l1_ratio):
        self.lambda_reg = lambda_reg
        self.l1_ratio = l1_ratio

    def __call__(self, theta):
        l1_contribution = self.l1_ratio * self.lambda_reg * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l1_ratio) * self.lambda_reg * 0.5 * np.sum(np.square(theta))
        return l1_contribution + l2_contribution

    def derivation(self, theta):
        l1_derivation = self.lambda_reg * self.l1_ratio * np.sign(theta)
        l2_derivation = self.lambda_reg * (1 - self.l1_ratio) * theta
        return l1_derivation + l2_derivation
