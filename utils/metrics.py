import numpy as np

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def classification_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    precisions = []
    recalls = []
    f1_scores = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores)
    }


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2) 
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) 
    
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)

def adjusted_r2_score(y_true, y_pred, num_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = num_features
    
    if n - p - 1 <= 0:
        return 0.0
        
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

