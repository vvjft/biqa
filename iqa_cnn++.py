import pandas as pd
import numpy as np
import os
import argparse

from data_loader import tid2013_loader, kadid10k_loader
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import curve_fit

# TO DO:
    # save results to txt
    # check if mapping is correct
    # create model without classification
    

def split_data(metadata, measureName, validation=True):  
    metadata = metadata.reset_index().rename(columns={'index': 'original_index'})
    metadata['image_id'] = metadata['image'].apply(lambda x: '_'.join(x.split('_')[:4]))
    metadata['original_index'] = metadata['original_index'].astype(int)
    
    groups = metadata.groupby('image_id').agg(list).reset_index()
    train, test = train_test_split(groups, test_size=0.2, random_state=40)
    meta_sets = []
    
    if validation:
        train, val = train_test_split(train, test_size=0.25, random_state=40)
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
    mos = np.sort(mos)
    dmos = np.random.choice(np.sort(dmos), size = len(mos), replace = False)
    params, _ = curve_fit(logistic_function, mos, dmos, p0=initial_params, maxfev=10000)

    return logistic_function(mos, *params)
    
def build_model(num_classes):
    inputs = layers.Input(shape=(32, 32, 1))
    x = layers.Conv2D(8, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    
    regression_output = layers.Dense(1, activation='linear', name='regression')(x)
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    model = models.Model(inputs=inputs, outputs=[regression_output, classification_output])
    return model
    
def show_results(meta_test, y_pred_reg, y_pred_class, measureName, distortion_mapping):
    
    def group_results(metadata, measureName):  
        metadata['prefix'] = metadata['image'].str.extract(r'(i\d+_\d+_\d+)_patch')
        
        grouped = metadata.groupby('prefix').agg(
            measure=(measureName, 'first'),  
            pred_measure=(f'pred_{measureName}', 'mean'),
            distortion=('distortion', 'first'),
            pred_distortion = ('pred_distortion', lambda x: x.mode().iloc[0])  
        ).reset_index()
              
        grouped.rename(columns={'prefix': 'image', 'measure': measureName, 'pred_measure': f'pred_{measureName}'}, inplace=True)
        return grouped
  
    sequential_mapping = {i: key for i, key in enumerate(sorted(distortion_mapping.keys()))}
    
    meta_test[measureName] = pd.to_numeric(meta_test[measureName], errors='coerce').astype('float32')
    meta_test['distortion'] = pd.to_numeric(meta_test['distortion'], errors='coerce').astype('int64')
    indices = np.argmax(y_pred_class, axis=1)
    meta_test[f'pred_{measureName}'] = y_pred_reg.flatten()
    meta_test['pred_distortion'] = indices
    meta_test['distortion'] = meta_test['distortion'].map(sequential_mapping).map(distortion_mapping)
    meta_test['pred_distortion'] = meta_test['pred_distortion'].map(sequential_mapping).map(distortion_mapping)
    meta_test.to_csv('meta_test.csv')
    results = group_results(meta_test, measureName)
    results.to_csv('results.csv')

    acc = accuracy_score(results['distortion'], results['pred_distortion'])
    lcc = pearsonr(results[f'pred_{measureName}'], results[measureName])[0]
    srocc = spearmanr(results[f'pred_{measureName}'], results[measureName])[0]
    krcc = kendalltau(results[f'pred_{measureName}'], results[measureName])[0]
    mae = mean_absolute_error(results[measureName], results[f'pred_{measureName}'])
    
    print('All:')
    print(f'  ACC (Accuracy): {acc}')
    print(f'  LCC (Linear Correlation Coefficient): {lcc}')
    print(f'  SROCC (Spearman Rank Order Correlation Coefficient): {srocc}')
    print(f'  KRCC (Kendall Rank Correlation Coefficient): {krcc}')
    print(f'  MAE (Mean Absolute Error): {mae}')
    
    distortions = results.groupby('distortion') 
    for name, distortion in distortions:
        group_accuracy_score = accuracy_score(distortion['distortion'], distortion['pred_distortion'])
        group_lcc = pearsonr(distortion[f'pred_{measureName}'], distortion[measureName])[0]
        group_srocc = spearmanr(distortion[f'pred_{measureName}'], distortion[measureName])[0]
        group_krcc = kendalltau(distortion[f'pred_{measureName}'], distortion[measureName])[0]
        group_mae = mean_absolute_error(distortion[measureName], distortion[f'pred_{measureName}'])
    
        print(f'{name}')
        print(f'  ACC: {group_accuracy_score}')
        print(f'  LCC: {group_lcc}')
        print(f'  SROCC: {group_srocc}')
        print(f'  KRCC: {group_krcc}')
        print(f'  MAE: {group_mae}')

def main(training, test):
    databases = {'tid2013': tid2013_loader, 'kadid10k': kadid10k_loader}
    
    training_loader = databases[training]()
    num_classes = training_loader.num_classes
    training_data = training_loader.metadata
    X = training_loader.X
    y_reg = training_loader.y_reg
    y_class = training_loader.y_class
    
    if test is None or test==training:
        measureName = training_loader.measureName
        meta_train, meta_val, meta_test  = split_data(training_data, measureName)
   
        train_indices, val_indices, test_indices = meta_train.index, meta_val.index, meta_test.index
        X_test =  X[test_indices]
        y_test_reg =  y_reg[test_indices]
        y_test_class =  y_class[test_indices]        
    else:
        test_loader = databases[test]()
        test_data = test_loader.metadata
        measureName = test_loader.measureName
        
        training_data[measureName] = mos2dmos(training_data['MOS'], test_data[measureName])
        meta_train, meta_val = split_data(training_data, 'MOS', False)
        meta_test = test_loader.metadata
        train_indices, val_indices = meta_train.index, meta_val.index
        
        X_test =  test_loader.X
        y_test_reg =  test_loader.y_reg
        y_test_class =  test_loader.y_class
           
    X_train, X_val = X[train_indices], X[val_indices] 
    y_train_reg, y_val_reg = y_reg[train_indices], y_reg[val_indices]
    y_train_class, y_val_class = y_class[train_indices], y_class[val_indices]
            
    model = build_model(num_classes)
    model.compile(optimizer='adam', loss=['mae', 'sparse_categorical_crossentropy'])
    model.fit(X_train, [y_train_reg,  y_train_class], epochs=1, batch_size=32, verbose=2)

    y_pred_reg, y_pred_class = model.predict(X_test, verbose=2)
    distortion_mapping = training_loader.distortion_mapping_kadid10k
    show_results(meta_test, y_pred_reg, y_pred_class, measureName, distortion_mapping)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description to be filled')
    choices = ['tid2013', 'kadid10k']
    
    parser.add_argument('--training', choices=choices, type=str, default='tid2013', help='Database for training set (default: tid2013)')
    parser.add_argument('--test', choices=choices, type=str, help='Database for test set. If not specified. Evaluation is done on trainig set.')
    args = parser.parse_args()

    main(args.training, args.test)

