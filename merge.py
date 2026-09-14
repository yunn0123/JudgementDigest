import pandas as pd
import re

files_prefix = 'judgments_ADV_TPD_M_'
files_suffix = ['20260913_224726.xlsx', '20260914_050629.xlsx']
final_name = '25_07-09.xlsx'
result = pd.DataFrame()

for file_suffix in files_suffix:
    file_path = files_prefix + file_suffix
    df = pd.read_excel(file_path)
    
    result = pd.concat([result, df], ignore_index=True)
    
    
result.to_excel(final_name, index=False)