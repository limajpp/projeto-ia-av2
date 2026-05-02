import numpy as np
import csv

def load_local_arff(filepath):
    print(f"loading ARFF file from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    data_lines = []
    headers = []
    is_data_section = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue 
        if line.upper().startswith('@ATTRIBUTE'):
            parts = line.split()
            if len(parts) >= 2:
                header_name = parts[1].replace("'", "").replace('"', '')
                headers.append(header_name)
        elif line.upper().startswith('@DATA'):
            is_data_section = True 
            continue
        elif is_data_section:
            data_lines.append(line)
            
    csv_reader = csv.reader(data_lines)
    
    final_matrix = []
    expected_cols = len(headers)
    
    for row in csv_reader:
        if len(row) != expected_cols:
            continue
        converted_row = []
        for item in row:
            item = item.strip()
            try:
                converted_row.append(float(item))
            except ValueError:
                if item == '' or item == '?':
                     converted_row.append(np.nan)
                else:
                     converted_row.append(item) 
        final_matrix.append(converted_row)
        
    return np.array(final_matrix, dtype=object), headers

