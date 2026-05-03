import numpy as np
import time

from utils.loader import load_local_arff
from utils.encoder import impute_and_clean_matrix, label_encode_column
from utils.splitter import k_fold_split
from utils.metrics import classification_metrics, r2_score, adjusted_r2_score

from models.knn import KNNClassifier, KNNRegressor
from models.naive_bayes import UnivariateNaiveBayes
from models.multivariate_bayes import MultivariateBayes
from models.linear_regression import MultipleLinearRegression

if __name__ == "__main__":
    ANIME_DATASET = r"C:\Users\HimeIceCream\Documents\Desenvolvimento\Unifor\IA\trabalhoAV2\data\dataset_ANIME" 
    STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS_DATASET = r"C:\Users\HimeIceCream\Documents\Desenvolvimento\Unifor\IA\trabalhoAV2\data\dataset_ STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS"
    K_FOLDS = 5

    print("="*60)
    print("CLASSIFICAÇÃO (Students Dropout And Academic Success Dataset)")
    print("="*60)
    
    try:
        clf_data, clf_columns = load_local_arff(STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS_DATASET)
        X_raw_clf = clf_data[:, :-1]
        y_raw_clf = clf_data[:, -1]
        
        X_clean_clf = impute_and_clean_matrix(X_raw_clf)
        y_encoded_clf, y_mapping = label_encode_column(y_raw_clf)
        
        print(f"Dataset carregado: {clf_data.shape[0]} amostras, {X_clean_clf.shape[1]} features.")
        print(f"Mapeamento do Target: {y_mapping}\n")

        folds_clf = k_fold_split(X_clean_clf, k=K_FOLDS)
        
        k_neighbors = 5
        for metric in ['euclidean', 'manhattan']:
            print(f"--> KNN Classifier | Distância: {metric.capitalize()} | K={k_neighbors}")
            fold_acc, fold_f1, fold_prec, fold_recall, fold_train_time, fold_test_time = [], [], [], [], [], []
            
            for train_idx, test_idx in folds_clf:
                X_train, y_train = X_clean_clf[train_idx], y_encoded_clf[train_idx]
                X_test, y_test = X_clean_clf[test_idx], y_encoded_clf[test_idx]
                
                model = KNNClassifier(k=k_neighbors, distance_metric=metric)
                
                start_train = time.time()
                model.fit(X_train, y_train)
                fold_train_time.append(time.time() - start_train)
                
                start_test = time.time()
                predictions = model.predict(X_test)
                fold_test_time.append(time.time() - start_test)
                
                metrics = classification_metrics(y_test, predictions)
                fold_acc.append(metrics['accuracy'])
                fold_prec.append(metrics['precision'])
                fold_recall.append(metrics['recall'])
                fold_f1.append(metrics['f1_score'])
                
            print(f"    Accuracy    : {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
            print(f"    Precision   : {np.mean(fold_prec):.4f} ± {np.std(fold_prec):.4f}")
            print(f"    Recall      : {np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}")
            print(f"    F1-Score    : {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
            print(f"    Train Time  : {np.mean(fold_train_time):.4f}s ± {np.std(fold_train_time):.4f}s")
            print(f"    Test Time   : {np.mean(fold_test_time):.4f}s ± {np.std(fold_test_time):.4f}s\n")

        print(f"--> Univariate Naive Bayes")
        fold_acc, fold_f1, fold_prec, fold_recall, fold_train_time, fold_test_time = [], [], [], [], [], []
        
        for train_idx, test_idx in folds_clf:
            X_train, y_train = X_clean_clf[train_idx], y_encoded_clf[train_idx]
            X_test, y_test = X_clean_clf[test_idx], y_encoded_clf[test_idx]
            
            model = UnivariateNaiveBayes()
            
            start_train = time.time()
            model.fit(X_train, y_train)
            fold_train_time.append(time.time() - start_train)
            
            start_test = time.time()
            predictions = model.predict(X_test)
            fold_test_time.append(time.time() - start_test)
            
            metrics = classification_metrics(y_test, predictions)
            fold_acc.append(metrics['accuracy'])
            fold_prec.append(metrics['precision'])
            fold_recall.append(metrics['recall'])
            fold_f1.append(metrics['f1_score'])
            
        print(f"    Accuracy    : {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
        print(f"    Precision   : {np.mean(fold_prec):.4f} ± {np.std(fold_prec):.4f}")
        print(f"    Recall      : {np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}")
        print(f"    F1-Score    : {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
        print(f"    Train Time  : {np.mean(fold_train_time):.4f}s ± {np.std(fold_train_time):.4f}s")
        print(f"    Test Time   : {np.mean(fold_test_time):.4f}s ± {np.std(fold_test_time):.4f}s\n")

        print(f"--> Multivariate Bayes")
        fold_acc, fold_f1, fold_prec, fold_recall, fold_train_time, fold_test_time = [], [], [], [], [], []
        
        for train_idx, test_idx in folds_clf:
            X_train, y_train = X_clean_clf[train_idx], y_encoded_clf[train_idx]
            X_test, y_test = X_clean_clf[test_idx], y_encoded_clf[test_idx]
            
            model = MultivariateBayes()
            
            start_train = time.time()
            model.fit(X_train, y_train)
            fold_train_time.append(time.time() - start_train)
            
            start_test = time.time()
            predictions = model.predict(X_test)
            fold_test_time.append(time.time() - start_test)
            
            metrics = classification_metrics(y_test, predictions)
            fold_acc.append(metrics['accuracy'])
            fold_prec.append(metrics['precision'])
            fold_recall.append(metrics['recall'])
            fold_f1.append(metrics['f1_score'])
            
        print(f"    Accuracy    : {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
        print(f"    Precision   : {np.mean(fold_prec):.4f} ± {np.std(fold_prec):.4f}")
        print(f"    Recall      : {np.mean(fold_recall):.4f} ± {np.std(fold_recall):.4f}")
        print(f"    F1-Score    : {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
        print(f"    Train Time  : {np.mean(fold_train_time):.4f}s ± {np.std(fold_train_time):.4f}s")
        print(f"    Test Time   : {np.mean(fold_test_time):.4f}s ± {np.std(fold_test_time):.4f}s\n")

    except Exception as e:
        print(f"ERRO NA CLASSIFICAÇÃO: {e}")

    print("="*60)
    print("REGRESSÃO (Anime Dataset)")
    print("="*60)
    
    try:
        reg_data, reg_columns = load_local_arff(ANIME_DATASET)
        X_raw_reg = reg_data[:, :-2]
        y_raw_reg = reg_data[:, -2].astype(float) 
        
        X_clean_reg = impute_and_clean_matrix(X_raw_reg)
        y_mean = np.nanmean(y_raw_reg)
        y_raw_reg[np.isnan(y_raw_reg)] = y_mean
        
        num_features = X_clean_reg.shape[1]
        
        print(f"Dataset carregado: {reg_data.shape[0]} amostras, {num_features} features.")
        print(f"Target: {reg_columns[-2]}\n")
        
        folds_reg = k_fold_split(X_clean_reg, k=K_FOLDS)

        for metric in ['euclidean', 'manhattan']:
            print(f"--> KNN Regressor | Distância: {metric.capitalize()} | K={k_neighbors}")
            fold_r2, fold_adj_r2, fold_train_time, fold_test_time = [], [], [], []
            
            for train_idx, test_idx in folds_reg:
                X_train, y_train = X_clean_reg[train_idx], y_raw_reg[train_idx]
                X_test, y_test = X_clean_reg[test_idx], y_raw_reg[test_idx]
                
                model = KNNRegressor(k=k_neighbors, distance_metric=metric)
                
                start_train = time.time()
                model.fit(X_train, y_train)
                fold_train_time.append(time.time() - start_train)
                
                start_test = time.time()
                predictions = model.predict(X_test)
                fold_test_time.append(time.time() - start_test)
                
                fold_r2.append(r2_score(y_test, predictions))
                fold_adj_r2.append(adjusted_r2_score(y_test, predictions, num_features))
                
            print(f"    R2 Score    : {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")
            print(f"    Adjusted R2 : {np.mean(fold_adj_r2):.4f} ± {np.std(fold_adj_r2):.4f}")
            print(f"    Train Time  : {np.mean(fold_train_time):.4f}s ± {np.std(fold_train_time):.4f}s")
            print(f"    Test Time   : {np.mean(fold_test_time):.4f}s ± {np.std(fold_test_time):.4f}s\n")

        print(f"--> Multiple Linear Regression (Closed-Form)")
        fold_r2, fold_adj_r2, fold_train_time, fold_test_time = [], [], [], []
        
        for train_idx, test_idx in folds_reg:
            X_train, y_train = X_clean_reg[train_idx], y_raw_reg[train_idx]
            X_test, y_test = X_clean_reg[test_idx], y_raw_reg[test_idx]
            
            model = MultipleLinearRegression()
            
            start_train = time.time()
            model.fit(X_train, y_train)
            fold_train_time.append(time.time() - start_train)
            
            start_test = time.time()
            predictions = model.predict(X_test)
            fold_test_time.append(time.time() - start_test)
            
            fold_r2.append(r2_score(y_test, predictions))
            fold_adj_r2.append(adjusted_r2_score(y_test, predictions, num_features))
            
        print(f"    R2 Score    : {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")
        print(f"    Adjusted R2 : {np.mean(fold_adj_r2):.4f} ± {np.std(fold_adj_r2):.4f}")
        print(f"    Train Time  : {np.mean(fold_train_time):.4f}s ± {np.std(fold_train_time):.4f}s")
        print(f"    Test Time   : {np.mean(fold_test_time):.4f}s ± {np.std(fold_test_time):.4f}s\n")

    except Exception as e:
        print(f"ERRO NA REGRESSÃO: {e}")

