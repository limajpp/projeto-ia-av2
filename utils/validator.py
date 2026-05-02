import numpy as np

def k_fold_split(X, k=5, shuffle=True, random_state=42):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
    fold_sizes = np.full(k, num_samples // k, dtype=int)
    fold_sizes[:num_samples % k] += 1
    
    current_index = 0
    folds = []
    
    for fold_size in fold_sizes:
        start, stop = current_index, current_index + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        
        folds.append((train_indices, test_indices))
        current_index = stop
        
    return folds

