import numpy as np
from utils.loader import load_local_arff

if __name__ == "__main__":
    # WARNING: Make sure you download the files and place them in the same folder as this script, or update these paths!
    # My datasets file paths:    
    ANIME = r"C:\Users\HimeIceCream\Documents\Desenvolvimento\Unifor\IA\trabalhoAV2\data\dataset_ANIME" 
    STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS = r"C:\Users\HimeIceCream\Documents\Desenvolvimento\Unifor\IA\trabalhoAV2\data\dataset_ STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS"

    reg_data, reg_columns = load_local_arff(ANIME)
    print(f"Features found: {len(reg_columns)}")
    print(f"Total instances: {reg_data.shape[0]}")
    print(f"Target column: {reg_columns[-1]}")

    reg_data, reg_columns = load_local_arff(STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS)
    print(f"Features found: {len(reg_columns)}")
    print(f"Total instances: {reg_data.shape[0]}")
    print(f"Target column: {reg_columns[-1]}")

