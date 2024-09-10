import pandas as pd
import numpy as np
import argparse
import os

from data_loader import tid2013_loader, kadid10k_loader
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import curve_fit

# TO DO:
    # verify mos2dmos mapping
    
def split_data(metadata, measureName, validation=True):  
    metadata = metadata.reset_index().rename(columns={'index': 'original_index'})
    metadata['image_id'] = metadata['image'].apply(lambda x: '_'.join(x.split('_')[:4]))
    metadata['original_index'] = metadata['original_index'].astype(int)
    
    groups = metadata.groupby('image_id').agg(list).reset_index()
    train, test = train_test_split(groups, test_size=0.2, stratify=groups['distortion'], random_state=42)
    meta_sets = []
    
    if validation:
        train, val = train_test_split(train, test_size=0.25, stratify=train['distortion'], random_state=42)
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
    if len(mos) > len(dmos):
        dmos = np.random.choice(mos, size=len(mos), replace=False)
    else:
        dmos = np.random.choice(np.sort(dmos), size=len(mos), replace=False)
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

def build_model2(): # no classification
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model
    
def show_results(meta_test, y_pred_reg, y_pred_class, measureName, distortion_mapping, classify=True):
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

    sequential_mapping = {i: key for i, key in enumerate(sorted(distortion_mapping.keys()))}
    
    meta_test[measureName] = pd.to_numeric(meta_test[measureName], errors='coerce').astype('float32')
    meta_test[f'pred_{measureName}'] = y_pred_reg.flatten()
    meta_test['pred_distortion'] = None
    if classify:
        meta_test['distortion'] = pd.to_numeric(meta_test['distortion'], errors='coerce').astype('int64')
        indices = np.argmax(y_pred_class, axis=1)      
        meta_test['pred_distortion'] = indices
        meta_test['distortion'] = meta_test['distortion'].map(sequential_mapping).map(distortion_mapping)
        meta_test['pred_distortion'] = meta_test['pred_distortion'].map(sequential_mapping).map(distortion_mapping)
    #meta_test.to_csv('meta_test.csv')
    results = group_results(meta_test, measureName)
    results.to_csv('results.csv')

    lcc = pearsonr(results[f'pred_{measureName}'], results[measureName])[0]
    srocc = spearmanr(results[f'pred_{measureName}'], results[measureName])[0]
    krcc = kendalltau(results[f'pred_{measureName}'], results[measureName])[0]
    mae = mean_absolute_error(results[measureName], results[f'pred_{measureName}'])
    if classify:
        acc = accuracy_score(results['distortion'], results['pred_distortion'])
    
    with open('results.txt', 'w') as file:
        file.write('All:\n')
        
        file.write(f'  LCC (Linear Correlation Coefficient): {lcc}\n')
        file.write(f'  SROCC (Spearman Rank Order Correlation Coefficient): {srocc}\n')
        file.write(f'  KRCC (Kendall Rank Correlation Coefficient): {krcc}\n')
        file.write(f'  MAE (Mean Absolute Error): {mae}\n')
        if classify:
            file.write(f'  ACC (Accuracy): {acc}\n')
            distortions = results.groupby('distortion') 
            for name, distortion in distortions:
                
                group_accuracy_score = accuracy_score(distortion['distortion'], distortion['pred_distortion'])
                group_lcc = pearsonr(distortion[f'pred_{measureName}'], distortion[measureName])[0]
                group_srocc = spearmanr(distortion[f'pred_{measureName}'], distortion[measureName])[0]
                group_krcc = kendalltau(distortion[f'pred_{measureName}'], distortion[measureName])[0]
                group_mae = mean_absolute_error(distortion[measureName], distortion[f'pred_{measureName}'])
            
                file.write(f'{name}:\n')
                file.write(f'  ACC: {group_accuracy_score}\n')
                file.write(f'  LCC: {group_lcc}\n')
                file.write(f'  SROCC: {group_srocc}\n')
                file.write(f'  KRCC: {group_krcc}\n')
                file.write(f'  MAE: {group_mae}\n')
    print(f'Results saved to results.txt')

def main(training, test, epochs, classify=True):
    databases = {'tid2013': tid2013_loader, 'kadid10k': kadid10k_loader}
    training_loader = databases[training](test, as_training=True)
    #training_loader = databases[training](training, as_training=False)
    num_classes = training_loader.num_classes
    training_data = training_loader.metadata
    X = training_loader.X
    y_reg = training_loader.y_reg
    y_class = training_loader.y_class
    
    if test==training:
        measureName = training_loader.measureName
        distortion_mapping = training_loader.distortion_mapping
        meta_train, meta_val, meta_test  = split_data(training_data, measureName)
        train_indices, val_indices, test_indices = meta_train.index, meta_val.index, meta_test.index
        X_test =  X[test_indices]
        y_test_reg =  y_reg[test_indices]
        y_test_class =  y_class[test_indices]        
    else:
        test_loader = databases[test](training)
        test_data = test_loader.metadata
        test_measureName = test_loader.measureName
        training_measureName = training_loader.measureName
        measureName = test_measureName
        distortion_mapping = getattr(training_loader, f'distortion_mapping_{test}')
        
        training_data[training_measureName] = mos2dmos(training_data[training_measureName], test_data[test_measureName])
        meta_train, meta_val = split_data(training_data, training_measureName, validation=False)
        #meta_train.to_csv('meta_train.csv')
        meta_test = test_loader.metadata
        train_indices, val_indices = meta_train.index, meta_val.index
        
        X_test =  test_loader.X
        y_test_reg =  test_loader.y_reg
        y_test_class =  test_loader.y_class
           
    X_train, X_val = X[train_indices], X[val_indices] 
    y_train_reg, y_val_reg = y_reg[train_indices], y_reg[val_indices]
    y_train_class, y_val_class = y_class[train_indices], y_class[val_indices]

    if classify:    
        model = build_model(num_classes)
        model.compile(optimizer='adam', loss=['mae', 'sparse_categorical_crossentropy'])
        model.fit(X_train, [y_train_reg,  y_train_class], epochs=epochs, batch_size=32, verbose=2)
        y_pred_reg, y_pred_class = model.predict(X_test, verbose=2)
    else:
        model = build_model2()
        model.compile(optimizer='adam', loss='mean_absolute_error')
        model.fit(X_train, y_train_reg, epochs=epochs, batch_size=32, verbose=2)
        y_pred_reg = model.predict(X_test, verbose=2)
        y_pred_class = None
  
    show_results(meta_test, y_pred_reg, y_pred_class, measureName, distortion_mapping, classify)
    #show_results2(meta_test, y_pred_reg,  measureName)

if __name__ == "__main__":
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser(description='Description to be filled')
    choices = ['tid2013', 'kadid10k']
    
    parser.add_argument('--training', choices=choices, type=str, default='tid2013', help='Database for training set (default: tid2013)')
    parser.add_argument('--test', choices=choices, type=str, help='Database for test set. If not specified - evaluation is done on trainig set.')
    parser.add_argument('--epochs', type=int, default = 1, help='Number of epochs (default: 1)')
    parser.add_argument('--no_classify', action='store_false', help='Disable classification of the distortion')
    args = parser.parse_args()
    if args.test == None:
        args.test = args.training
    main(args.training, args.test, args.epochs, args.no_classify)

