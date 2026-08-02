import os
import pandas as pd

base = r'c:\Users\Admin\Downloads\LM_462'
paths = [
    os.path.join(base, 'Activity 1 & 6 & 10 (Dataset)', 'iris.csv'),
    os.path.join(base, '‏‏Activity 2 & 3 (Dataset)', 'foodtruck.csv'),
    os.path.join(base, '‏‏Activity 4 (Dataset)', 'Breast_cancer.csv'),
    os.path.join(base, '‏‏Activity 5 & 8 (Dataset)', 'pima_indian_diabeties.csv'),
    os.path.join(base, '‏‏Activity 7 (Dataset)', 'golf-dataset.csv'),
]

for path in paths:
    print('FILE', os.path.basename(path))
    try:
        df = pd.read_csv(path)
        print(df.head(5).to_string(index=False))
        print('columns:', df.columns.tolist())
        print('shape:', df.shape)
    except Exception as e:
        print('ERROR', e)
    print('---')
