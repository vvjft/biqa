import numpy as np
import pandas as pd
import os
import datetime
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_absolute_error

def split_data(metadata, measureName, validation=True):  
    metadata = metadata.reset_index().rename(columns={'index': 'original_index'})
    metadata['image_id'] = metadata['image'].apply(lambda x: '_'.join(x.split('_')[:4]))
    metadata['original_index'] = metadata['original_index'].astype(int)
    
    groups = metadata.groupby('image_id').agg(list).reset_index()
    train, test = train_test_split(groups, test_size=0.2, stratify = groups['distortion'], random_state=42)
    meta_sets = []
    
    if validation:
        train, val = train_test_split(train, test_size=0.25, stratify = train['distortion'], random_state=42)
        datasets = [train, val, test]
    else:
        datasets = [train, test]
        
    for dataset in datasets:
        meta_set = dataset.explode(['image', measureName, 'distortion', 'original_index'])
        meta_set['original_index'] = meta_set['original_index'].astype(int) 
        meta_set.set_index('original_index', inplace=True) 
        meta_sets.append(meta_set)
        
    return meta_sets 

def mos2dmos(mos, dmos):
    '''
    Function that maps one measure score into the other.
    Note: not necessarily mos to dmos. Can be dmos to mos or mos to mos
    '''
    def logistic_function(x, a, b, c, d):
        return a / (1 + np.exp(-c * (x - d))) + b

    initial_params = [0, 0, 0, np.median(mos)]
    sorted_mos = np.sort(mos)
    sorted_dmos = np.sort(dmos)
    if len(mos) > len(dmos):
        sorted_mos = np.sort(np.random.choice(mos, size=len(dmos), replace=False))
    else:
        sorted_dmos = np.sort(np.random.choice(dmos, size=len(mos), replace=False))
    params, _ = curve_fit(logistic_function, sorted_mos, sorted_dmos, p0=initial_params, maxfev=10000)
    return logistic_function(mos, *params)

def evaluate(meta_test, y_pred_reg, measureName, distortion_mapping, classify=False):
    def group_results(metadata, measureName):
        metadata['prefix'] = metadata['image'].str.extract(r'(i\d+_\d+_\d+)_patch')

        grouped = metadata.groupby('prefix').agg(
            measure=(measureName, 'first'),  
            pred_measure=(f'pred_{measureName}', 'mean'),
            distortion=('distortion', 'first'),
            pred_distortion = ('pred_distortion', lambda x: x.mode().iloc[0] if x.notna().any() else None)  
        ).reset_index()

        grouped.rename(columns={'prefix': 'image', 'measure': measureName, 'pred_measure': f'pred_{measureName}'}, inplace=True)
        return grouped
 
    if classify:
        # when single database is tested, distortion labels start from 1
        sequential_mapping = {i+1: key for i, key in enumerate(sorted(distortion_mapping.keys()))}
    else:
        # when performing cross databse test, we sequentialize labels (starting from 0)
        sequential_mapping = {i: key for i, key in enumerate(sorted(distortion_mapping.keys()))}
    print(meta_test.shape)
    meta_test[measureName] = pd.to_numeric(meta_test[measureName], errors='coerce').astype('float16')
    meta_test[f'pred_{measureName}'] = y_pred_reg.flatten()  
    meta_test['distortion'] = pd.to_numeric(meta_test['distortion'], errors='coerce').fillna(0).astype('int16')
    meta_test['distortion'] = meta_test['distortion'].map(sequential_mapping).map(distortion_mapping)
    meta_test['pred_distortion'] = None
    

    results = group_results(meta_test, measureName)
    results.to_csv('results.csv')

    lcc = pearsonr(results[f'pred_{measureName}'], results[measureName])[0]
    srocc = spearmanr(results[f'pred_{measureName}'], results[measureName])[0]
    krcc = kendalltau(results[f'pred_{measureName}'], results[measureName])[0]
    mae = mean_absolute_error(results[measureName], results[f'pred_{measureName}'])

    

    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    folder_path = './wyniki'
    os.makedirs(folder_path, exist_ok=True)
    filename = f"results_{timestamp}.txt"
    filepath = os.path.join(folder_path, filename)
    with open(filepath, 'w') as file:
        file.write('All:\n')   
        file.write(f'  PLCC (Pearson Linear Correlation Coefficient): {lcc}\n')
        file.write(f'  SROCC (Spearman Rank Order Correlation Coefficient): {srocc}\n')
        file.write(f'  KRCC (Kendall Rank Correlation Coefficient): {krcc}\n')
        file.write(f'  MAE (Mean Absolute Error): {mae}\n')

    print(f'Results saved to {filepath}')
    return lcc