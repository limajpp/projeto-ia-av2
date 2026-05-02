import numpy as np

def label_encode_column(column_data):
    valid_data = [val for val in column_data if isinstance(val, str)]
    unique_categories = sorted(list(set(valid_data)))
    
    encoding_map = {val: idx for idx, val in enumerate(unique_categories)}
    
    encoded_col = []
    for val in column_data:
        if isinstance(val, str) and val in encoding_map:
            encoded_col.append(float(encoding_map[val]))
        else:
            encoded_col.append(np.nan) 
            
    return np.array(encoded_col), encoding_map

def impute_and_clean_matrix(X_matrix):
    num_cols = X_matrix.shape[1]
    cleaned_columns = []
    
    for col_idx in range(num_cols):
        col_data = X_matrix[:, col_idx]
        
        has_strings = any(isinstance(x, str) for x in col_data)
        if has_strings:
            sample_str = next(x for x in col_data if isinstance(x, str))
            if len(sample_str) > 50 or sample_str.startswith('['):
                continue 
            encoded_col, _ = label_encode_column(col_data)
            cleaned_columns.append(encoded_col)
        else:
            cleaned_columns.append(col_data.astype(float))
            
    X_numeric = np.column_stack(cleaned_columns)
    for col_idx in range(X_numeric.shape[1]):
        col_data = X_numeric[:, col_idx]
        nan_mask = np.isnan(col_data)
        if np.any(nan_mask):
            col_mean = np.nanmean(col_data)
            col_data[nan_mask] = col_mean
        X_numeric[:, col_idx] = col_data
            
    return X_numeric

