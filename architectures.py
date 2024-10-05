import numpy as np
import pandas as pd
import os

from tensorflow.keras import layers, models
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import curve_fit
from sklearn.metrics import accuracy_score, mean_absolute_error

def build_model(num_classes):
    inputs = layers.Input(shape=(32, 32, 1))
    x = layers.Conv2D(8, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    
    regression_output = layers.Dense(1, activation='linear', name='regression_output')(x)
    classification_output = layers.Dense(num_classes, activation='softmax', name='classification_output')(x)
    model = models.Model(inputs=inputs, outputs=[regression_output, classification_output])
    return model

def build_model2(): # no classification
    inputs = layers.Input(shape=(32, 32, 1))
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    regression_output = layers.Dense(1, activation='linear', name='regression_output')(x)
    model = models.Model(inputs=inputs, outputs=regression_output)
    return model
import tensorflow as tf
def custom_loss(class_weight):
    def weighted_loss(y_train, y_pred_reg):
        y_train_class = tf.cast(y_train[:, 0], tf.int32)  # Convert class labels to integers
        y_train_reg = y_train[:, 1:]  # The rest are regression targets
        
        # Compute the Mean Absolute Error (MAE) loss using NumPy arrays
        mae_loss = tf.reduce_mean(tf.abs(y_train_reg - y_pred_reg), axis=-1)
        print(y_train_class)
        weight = tf.gather(class_weight, y_train_class)
        
        # Apply the class weights based on the class labels
        weighted_mae_loss = mae_loss * weight
        return weighted_mae_loss
    return weighted_loss

def evaluate(meta_test, y_pred_reg, y_pred_class, measureName, distortion_mapping, single, classify=False):
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
 
    if single:
        # when single database is tested, distortion labels start from 1
        sequential_mapping = {i+1: key for i, key in enumerate(sorted(distortion_mapping.keys()))}
    else:
        # when performing cross databse test, we sequentialize labels (starting from 0)
        sequential_mapping = {i: key for i, key in enumerate(sorted(distortion_mapping.keys()))}
    
    meta_test[measureName] = pd.to_numeric(meta_test[measureName], errors='coerce').astype('float32')
    meta_test[f'pred_{measureName}'] = y_pred_reg.flatten()  
    meta_test['distortion'] = pd.to_numeric(meta_test['distortion'], errors='coerce').fillna(0).astype('int64')
    meta_test['distortion'] = meta_test['distortion'].map(sequential_mapping).map(distortion_mapping)
    meta_test['pred_distortion'] = None
    
    if classify:
        indices = np.argmax(y_pred_class, axis=1)      
        meta_test['pred_distortion'] = indices 
        meta_test['pred_distortion'] = meta_test['pred_distortion'].map(sequential_mapping).map(distortion_mapping)
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
        file.write(f'  PLCC (Pearson Linear Correlation Coefficient): {lcc}\n')
        file.write(f'  SROCC (Spearman Rank Order Correlation Coefficient): {srocc}\n')
        file.write(f'  KRCC (Kendall Rank Correlation Coefficient): {krcc}\n')
        file.write(f'  MAE (Mean Absolute Error): {mae}\n')
        if classify:
            file.write(f'  ACC (Accuracy): {acc}\n')
        distortions = results.groupby('distortion')
        for name, distortion in distortions:
            if classify:
                group_accuracy_score = accuracy_score(distortion['distortion'], distortion['pred_distortion'])
            group_lcc = pearsonr(distortion[f'pred_{measureName}'], distortion[measureName])[0]
            group_srocc = spearmanr(distortion[f'pred_{measureName}'], distortion[measureName])[0]
            group_krcc = kendalltau(distortion[f'pred_{measureName}'], distortion[measureName])[0]
            group_mae = mean_absolute_error(distortion[measureName], distortion[f'pred_{measureName}'])
        
            file.write(f'{name}:\n')
            file.write(f'  PLCC: {group_lcc}\n')
            file.write(f'  SROCC: {group_srocc}\n')
            file.write(f'  KRCC: {group_krcc}\n')
            file.write(f'  MAE: {group_mae}\n')
            if classify:
                file.write(f'  ACC: {group_accuracy_score}\n')
    print(f'Results saved to results.txt')
    return lcc