import numpy as np

class MultivariateBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_cov_inv = {} 
        self.class_cov_det = {} 

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        for c in self.classes:
            X_c = X[y == c]
            
            self.class_priors[c] = X_c.shape[0] / n_samples
            
            self.class_means[c] = np.mean(X_c, axis=0)
            
            cov_matrix = np.cov(X_c, rowvar=False)
            
            epsilon = 1e-6
            np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + epsilon)
            
            self.class_cov_inv[c] = np.linalg.pinv(cov_matrix)
            
            _, logdet = np.linalg.slogdet(cov_matrix)
            self.class_cov_det[c] = logdet

    def _calculate_log_likelihood(self, x, mean, cov_inv, cov_logdet):
        d = len(x)
        diff = x - mean
        
        mahalanobis_dist = np.dot(np.dot(diff.T, cov_inv), diff)
        
        log_likelihood = -0.5 * (cov_logdet + mahalanobis_dist + d * np.log(2 * np.pi))
        return log_likelihood

    def predict(self, X_test):
        predictions = []
        
        for x_sample in X_test:
            class_posteriors = {}
            
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                log_likelihood = self._calculate_log_likelihood(
                    x_sample, 
                    self.class_means[c], 
                    self.class_cov_inv[c], 
                    self.class_cov_det[c]
                )
                
                class_posteriors[c] = prior + log_likelihood
                
            best_class = max(class_posteriors, key=class_posteriors.get)
            predictions.append(best_class)
            
        return np.array(predictions)
    
