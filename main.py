import numpy as np
from utils.loader import load_local_arff
from utils.encoder import label_encode_column, impute_and_clean_matrix
from utils.splitter import k_fold_split

if __name__ == "__main__":
     # WARNING: Make sure you download the files and place them in the same folder as this script, or update these paths!
    # My datasets file paths:    
    ANIME = r"C:\Users\HimeIceCream\Documents\Desenvolvimento\Unifor\IA\trabalhoAV2\data\dataset_ANIME" 
    STUDENTS = r"C:\Users\HimeIceCream\Documents\Desenvolvimento\Unifor\IA\trabalhoAV2\data\dataset_ STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS"

    print("\n" + "="*50)
    print("--- Testing Regression (Anime) with K-Fold ---")
    try:
        reg_data, reg_columns = load_local_arff(ANIME)
        X_raw_reg = reg_data[:, :-2]
        y_raw_reg = reg_data[:, -2].astype(float) 
        
        print("Cleaning Anime feature matrix...")
        X_clean_reg = impute_and_clean_matrix(X_raw_reg)
        
        y_mean = np.nanmean(y_raw_reg)
        y_raw_reg[np.isnan(y_raw_reg)] = y_mean
        
        print(f"Original shape: {reg_data.shape}")
        print(f"Cleaned X shape: {X_clean_reg.shape}")
        print(f"Target column used: {reg_columns[-2]}")
        
        K = 5
        print(f"Performing {K}-Fold Cross-Validation Split...")
        reg_folds = k_fold_split(X_clean_reg, k=K)
        print(f"Success! Fold 1 Train size: {len(reg_folds[0][0])}, Test size: {len(reg_folds[0][1])}")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")

    print("\n" + "="*50)
    print("--- Testing Classification (Students) with K-Fold ---")
    try:
        clf_data, clf_columns = load_local_arff(STUDENTS)
        X_raw_clf = clf_data[:, :-1]
        y_raw_clf = clf_data[:, -1]
        
        print("Cleaning Students feature matrix...")
        X_clean_clf = impute_and_clean_matrix(X_raw_clf)
        
        print("Encoding Target column...")
        y_encoded_clf, y_mapping = label_encode_column(y_raw_clf)
        
        print(f"Original shape: {clf_data.shape}")
        print(f"Cleaned X shape: {X_clean_clf.shape}")
        print(f"Target column: {clf_columns[-1]}")
        print(f"Target mapping: {y_mapping}")

        K = 5
        print(f"Performing {K}-Fold Cross-Validation Split...")
        clf_folds = k_fold_split(X_clean_clf, k=K)
        print(f"Success! Fold 1 Train size: {len(clf_folds[0][0])}, Test size: {len(clf_folds[0][1])}")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")

