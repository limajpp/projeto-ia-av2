import numpy as np

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, x_test_sample):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x_test_sample) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x_test_sample), axis=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def predict(self, X_test):
        predictions = []
        for x_test_sample in X_test:
            distances = self._compute_distances(x_test_sample)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            most_common = np.bincount(k_nearest_labels.astype(int)).argmax()
            predictions.append(most_common)
            
        return np.array(predictions)


class KNNRegressor:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, x_test_sample):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x_test_sample) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x_test_sample), axis=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def predict(self, X_test):
        predictions = []
        for x_test_sample in X_test:
            distances = self._compute_distances(x_test_sample)
            k_indices = np.argsort(distances)[:self.k]
            
            k_nearest_targets = self.y_train[k_indices]
            
            mean_value = np.mean(k_nearest_targets)
            predictions.append(mean_value)
            
        return np.array(predictions)
    
