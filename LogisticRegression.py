import numpy as np
import warnings
from math import sqrt

class LogisticRegression:
    """
    Multinomial Logistic Regression class with regularization, and various classification metrics.
    
    Features:
    - Implements multinomial (one-vs-rest) logistic regression.
    - L2 (Ridge) regularization.
    - Zeros or Xavier weight initialization.
    - Implements accuracy, precision, recall, and F1-score (per-class, macro, and weighted).
    Note: Features should be scaled before fitting.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, init_method='xavier', 
                 penalty='none', lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_method = init_method
        self.penalty = penalty
        self.lambda_reg = lambda_reg
        self.weights = None
        self.n_classes = None
        self.history = {'loss': []}

    def _initialize_weights(self, n_features, n_classes):
        """Initializes weights using zeros or Xavier method."""
        if self.init_method == 'xavier':
            limit = np.sqrt(6.0 / (n_features + n_classes))
            self.weights = np.random.uniform(-limit, limit, size=(n_features, n_classes))
        else: # 'zeros'
            self.weights = np.zeros((n_features, n_classes))

    def _softmax(self, z):
        """Applies the softmax function for multinomial classification."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, n_classes):
        """
        Fits the model to the training data using gradient descent.
        """
        n_samples, n_features = X.shape
        self.n_classes = n_classes
        self._initialize_weights(n_features, n_classes)

        # One-hot encode the labels
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        for i in range(self.max_iter):
            # Calculate linear model and apply softmax
            z = np.dot(X, self.weights)
            y_pred = self._softmax(z)

            # Calculate loss (Cross-Entropy with L2 penalty)
            loss = -np.sum(y_one_hot * np.log(y_pred + 1e-9)) / n_samples
            if self.penalty == 'ridge':
                loss += (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights**2)
            self.history['loss'].append(loss)

            # Calculate gradient (with L2 penalty)
            gradient = np.dot(X.T, (y_pred - y_one_hot)) / n_samples
            if self.penalty == 'ridge':
                gradient += (self.lambda_reg / n_samples) * self.weights

            # Update weights
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        """Predicts class labels for new data."""
        z = np.dot(X, self.weights)
        y_pred = self._softmax(z)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        """Predicts class probabilities for new data."""
        z = np.dot(X, self.weights)
        return self._softmax(z)

    def _confusion_matrix(self, y_true, y_pred):
        """Helper to compute a confusion matrix for all metrics."""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label, pred_label] += 1
        return cm

    def accuracy(self, y_true, y_pred):
        """Computes the overall accuracy."""
        return np.mean(y_true == y_pred)

    def precision(self, y_true, y_pred, class_label):
        """Computes precision for a single class."""
        cm = self._confusion_matrix(y_true, y_pred)
        tp = cm[class_label, class_label]
        fp = np.sum(cm[:, class_label]) - tp
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def recall(self, y_true, y_pred, class_label):
        """Computes recall for a single class."""
        cm = self._confusion_matrix(y_true, y_pred)
        tp = cm[class_label, class_label]
        fn = np.sum(cm[class_label, :]) - tp
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def f1_score(self, y_true, y_pred, class_label):
        """Computes F1-score for a single class."""
        prec = self.precision(y_true, y_pred, class_label)
        rec = self.recall(y_true, y_pred, class_label)
        return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def macro_precision(self, y_true, y_pred):
        """Computes macro-averaged precision."""
        precisions = [self.precision(y_true, y_pred, c) for c in range(self.n_classes)]
        return np.mean(precisions)

    def macro_recall(self, y_true, y_pred):
        """Computes macro-averaged recall."""
        recalls = [self.recall(y_true, y_pred, c) for c in range(self.n_classes)]
        return np.mean(recalls)

    def macro_f1(self, y_true, y_pred):
        """Computes macro-averaged F1-score."""
        f1s = [self.f1_score(y_true, y_pred, c) for c in range(self.n_classes)]
        return np.mean(f1s)

    def weighted_precision(self, y_true, y_pred):
        """Computes weighted-averaged precision."""
        precisions = [self.precision(y_true, y_pred, c) for c in range(self.n_classes)]
        class_counts = np.bincount(y_true, minlength=self.n_classes)
        total_samples = len(y_true)
        weights = class_counts / total_samples
        return np.sum(np.array(precisions) * weights)

    def weighted_recall(self, y_true, y_pred):
        """Computes weighted-averaged recall."""
        recalls = [self.recall(y_true, y_pred, c) for c in range(self.n_classes)]
        class_counts = np.bincount(y_true, minlength=self.n_classes)
        total_samples = len(y_true)
        weights = class_counts / total_samples
        return np.sum(np.array(recalls) * weights)

    def weighted_f1(self, y_true, y_pred):
        """Computes weighted-averaged F1-score."""
        f1s = [self.f1_score(y_true, y_pred, c) for c in range(self.n_classes)]
        class_counts = np.bincount(y_true, minlength=self.n_classes)
        total_samples = len(y_true)
        weights = class_counts / total_samples
        return np.sum(np.array(f1s) * weights)