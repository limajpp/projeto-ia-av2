import numpy as np

class UnivariateNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_stds = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = X.shape[0] 
        
        for c in self.classes:
            X_c = X[y == c]
            
            self.class_priors[c] = X_c.shape[0] / n_samples
            
            self.class_means[c] = np.mean(X_c, axis=0)
            
            self.class_stds[c] = np.std(X_c, axis=0) + 1e-9 

    def _calculate_likelihood(self, x, mean, std):
        exponent = np.exp(-0.5 * ((x - mean) / std) ** 2)
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, X_test):
        predictions = []
        
        for x_sample in X_test:
            class_posteriors = {}
            
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                
                likelihoods = self._calculate_likelihood(x_sample, self.class_means[c], self.class_stds[c])
                
                log_likelihood = np.sum(np.log(likelihoods + 1e-9))
                
                posterior = prior + log_likelihood
                class_posteriors[c] = posterior
            
            best_class = max(class_posteriors, key=class_posteriors.get)
            predictions.append(best_class)
            
        return np.array(predictions)
    
