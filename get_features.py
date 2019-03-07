import pandas as pd
import numpy as np
from constants import ENZYME_COLS, DUMMY_COLS

def get_num_cols(df):
    '''Get columns where data is numeric'''
    return df.select_dtypes(
        include=[float, int, bool, np.dtype('int32'), np.dtype('int64'),
                 np.dtype('float32'), np.dtype('float64'), np.dtype('uint8'), 'float64']
    ).columns

COMMON_DRUGS = [
    'acetaminophen', 'aspirin', 'amiodarone', 'methimazole', 'simvastatin',
    'propythiouracil', 'barbiturates', 'cholestyramine', 'carbamazepine', 'aspirin'
]

def get_features():
    data = (pd.read_csv('warfarin_data/warfarin.csv').dropna(how='all', axis=1)
            .dropna(subset=['Age', 'Therapeutic Dose of Warfarin'])
            .dropna(how='all')
            .rename(columns={"Therapeutic Dose of Warfarin": "warfarin"}))
    data['enzyme_sum'] = data[ENZYME_COLS].fillna(0).sum(1).clip(1, None).values
    data['Age'] = data['Age'].str.partition('-')[0].str.strip('+').astype(int)
    data['Comorbidities'] = data['Comorbidities'].str.lower().str.strip()
    for drug in COMMON_DRUGS:
        data[f'taking_{drug}'] = data['Medications'].str.contains(f'; {drug}').fillna(False)
    feature_df = pd.get_dummies(data, columns=DUMMY_COLS)
    print(feature_df.columns)
    target = feature_df['warfarin']
    num_cols = get_num_cols(feature_df).drop('warfarin')
    return feature_df[num_cols].fillna(-1), target
