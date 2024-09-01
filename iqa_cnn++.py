import pandas as pd
import numpy as np
import os

from data_loader import tid2013_loader, kadid10k_loader
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

def split_data(metadata, measureName, validation=True):  
    metadata = metadata.reset_index().rename(columns={'index': 'original_index'})
    metadata['image_id'] = metadata['image'].apply(lambda x: '_'.join(x.split('_')[:4]))
    metadata['original_index'] = metadata['original_index'].astype(int)
    
    groups = metadata.groupby('image_id').agg(list).reset_index()
    train, test = train_test_split(groups, test_size=0.2, random_state=40)
    
    if validation:
        train, val = train_test_split(train, test_size=0.25, random_state=40)
        datasets = [train, val, test]
    else:
        datasets = [train, test]
        
    meta_sets = []
    for dataset in datasets:
        meta_set = dataset.explode(['image', measureName, 'distortion', 'original_index'])
        meta_set['original_index'] = meta_set['original_index'].astype(int) 
        meta_set.set_index('original_index', inplace=True) 
        meta_sets.append(meta_set)
    return meta_sets 

def mos2dmos(mos, dmos):
    def logistic_function(x, a, b, c, d):
        return a / (1 + np.exp(-c * (x - d))) + b

    def get_logistic_fun(data_tid, data_live):
      def convert_mos_to_dmos(mos, dmos):
          initial_params = [0, 0, 0, np.median(mos)]  # initial guess for parameters
          popt, _ = curve_fit(logistic_function, mos, dmos, p0=initial_params, maxfev=10000)
          return popt
    
      mos = np.sort(mos)
      dmos = np.sort(dmos)
      params = convert_mos_to_dmos(mos_train, dmos_train)

      return logistic_function(mos)
    
def build_model(num_classes):
    inputs = layers.Input(shape=(32, 32, 1))
    x = layers.Conv2D(8, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    
    regression_output = layers.Dense(1, activation='linear')(x)
    classification_output = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=[regression_output, classification_output])
    return model

def group_results(metadata, measureName):
    metadata['prefix'] = metadata['image'].str.extract(r'(^.*_i\d+_\d+_\d+)_patch')

    grouped = metadata.groupby('prefix').agg(
        measure=(measureName, 'first'),  
        pred_measure=(f'pred_{measureName}', 'mean'),
        distortion=('distortion', 'first'),
        pred_distortion = ('pred_distortion', lambda x: x.mode().iloc[0])  
    ).reset_index()

    grouped.rename(columns={'prefix': 'image', 'measure': measureName, 'pred_measure': f'pred_{measureName}'}, inplace=True)
    return grouped
    
def main():
    cross = True
    if not cross:
        data_loader = tid2013_loader()
        num_classes = data_loader.num_classes
        measureName = data_loader.measureName
        metadata = data_loader.metadata
        meta_train, meta_val, meta_test  = split_data(metadata, measureName)

        X = data_loader.X
        y_reg = data_loader.y_reg
        y_class = data_loader.y_class
        
        train_indices, val_indices, test_indices = meta_train.index, meta_val.index, meta_test.index
        X_test =  X[test_indices]
        y_test_reg =  y_reg[test_indices]
        y_test_class =  y_class[test_indices]
        
    else:
        training_loader = tid2013_loader()
        test_loader = kadid10k_loader()
        num_classes = training_loader.num_classes
        measureName = test_loader.measureName
        training_data = training_loader.metadata
        test_data = test_loader.metadata
        training_data[measureName] = mos2dmos(training_data['MOS'], test_data[measureName])
        meta_train, meta_val = split_data(training_data, 'MOS', False)
        meta_test = test_loader.metadata
        
        X = training_loader.X
        y_reg = training_loader.y_reg
        y_class = training_loader.y_class
        
        train_indices, val_indices = meta_train.index, meta_val.index
        X_test =  test_loader.X
        y_test_reg =  test_loader.y_reg
        y_test_class =  test_loader.y_class
        
    X_train, X_val = X[train_indices], X[val_indices] 
    y_train_reg, y_val_reg = y_reg[train_indices], y_reg[val_indices]
    y_train_class, y_val_class = y_class[train_indices], y_class[val_indices]
            
    model = build_model(num_classes)
    model.compile(optimizer='adam', loss=['mae', 'sparse_categorical_crossentropy'])
    model.fit(X_train, [y_train_reg,  y_train_class], epochs=10, batch_size=32, verbose=2)

    y_pred_reg, y_pred_class = model.predict(X_test, verbose=2)

    meta_test[measureName] = pd.to_numeric(meta_test[measureName], errors='coerce').astype('float32')
    meta_test['distortion'] = pd.to_numeric(meta_test['distortion'], errors='coerce').astype('int64')
    indices = np.argmax(y_pred_class, axis=1)
    meta_test[f'pred_{measureName}'] = y_pred_reg.flatten()
    meta_test['pred_distortion'] = indices
    meta_test.to_csv('meta_test.csv')
    results = group_results(meta_test, measureName)
    results.to_csv('results.csv')

    print(accuracy_score(results['distortion'], results['pred_distortion']))
    
if __name__ == "__main__":
    main()
