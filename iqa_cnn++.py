import pandas as pd
import numpy as np
import os

from data_loader import tid2013_loader, kadid10k_loader
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

def split_data(metadata, measureName):  
    metadata = metadata.reset_index().rename(columns={'index': 'original_index'})
    metadata['image_id'] = metadata['image'].apply(lambda x: '_'.join(x.split('_')[:4]))
    metadata['original_index'] = metadata['original_index'].astype(int)
    
    groups = metadata.groupby('image_id').agg(list).reset_index()
    
    train, test = train_test_split(groups, test_size=0.2, random_state=40)
    train, val = train_test_split(train, test_size=0.25, random_state=40)
    
    meta_train = train.explode(['image', measureName, 'distortion', 'original_index'])
    meta_val = val.explode(['image', measureName, 'distortion', 'original_index'])
    meta_test = test.explode(['image', measureName, 'distortion', 'original_index'])

    meta_train['original_index'] = meta_train['original_index'].astype(int)
    meta_val['original_index'] = meta_val['original_index'].astype(int)
    meta_test['original_index'] = meta_test['original_index'].astype(int)
    
    meta_train.set_index('original_index', inplace=True)
    meta_val.set_index('original_index', inplace=True)
    meta_test.set_index('original_index', inplace=True)
     
    return meta_train, meta_val, meta_test

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
    num_classes = 26
    data_loader = kadid10k_loader()
    metadata = data_loader.metadata
    measureName = data_loader.measureName
    meta_train, meta_val, meta_test = split_data(metadata, measureName)

    X = data_loader.X
    y_reg = data_loader.y_reg
    y_class = data_loader.y_class
    train_indices, val_indices, test_indices = meta_train.index, meta_val.index, meta_test.index
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train_reg, y_val_reg, y_test_reg = y_reg[train_indices], y_reg[val_indices], y_reg[test_indices]
    y_train_class, y_val_class, y_test_class = y_class[train_indices], y_class[val_indices], y_class[test_indices]

    model = build_model(num_classes)
    model.compile(optimizer='adam', loss=['mae', 'sparse_categorical_crossentropy'])
    model.fit(X_train, [y_train_reg,  y_train_class], epochs=1, batch_size=32, verbose=2)

    y_pred_reg, y_pred_class = model.predict(X_test, verbose=2)

    meta_test[measureName] = pd.to_numeric(meta_test[measureName], errors='coerce').astype('float32')
    meta_test['distortion'] = pd.to_numeric(meta_test['distortion'], errors='coerce').astype('int64')
    indices = np.argmax(y_pred_class, axis=1)
    meta_test[f'pred_{measureName}'] = y_pred_reg.flatten()
    meta_test['pred_distortion'] = indices
    meta_test.to_csv('meta_test.csv')
    results = group_results(meta_test, data_loader.measureName)
    results.to_csv('results.csv')

    print(accuracy_score(results['distortion'], results['pred_distortion']))
    
if __name__ == "__main__":
    main()
