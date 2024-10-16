import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime

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

def build_model0(n_neurons1, n_neurons2, dropout_rate): # no classification

    inputs = layers.Input(shape=(32, 32, 1))
    
    x = layers.Conv2D(50, (7, 7), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x =  layers.Dense(n_neurons1, activation='relu')(x)
    x =  layers.Dense(n_neurons2, activation='relu')(x)
    x =  layers.Dropout(dropout_rate)(x)
    regression_output = layers.Dense(1, activation='linear', name='regression_output')(x)
    model = models.Model(inputs=inputs, outputs=regression_output)
    return model

def build_model_82(n_neruons=512, dropout_rate=0.5): # no classification
    model = models.Sequential([
        layers.Input(shape=(32, 32, 1)),
         
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(n_neruons, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(1, activation='linear')  
    ])
    return model

def custom_loss():
    def weighted_loss(y_true, y_pred):
        y_true_class = tf.cast(y_true[:, 0], tf.int32)  
        y_true_reg = tf.reshape(y_true[:, 1], tf.shape(y_pred))
        mae_loss = tf.abs(y_true_reg - y_pred)

        mask_14 = tf.cast(tf.equal(y_true_class, 14), tf.float32) * 5 
        mask_15 = tf.cast(tf.equal(y_true_class, 15), tf.float32) * 3  
        mask_16 = tf.cast(tf.equal(y_true_class, 16), tf.float32) * 7  
        mask_17 = tf.cast(tf.equal(y_true_class, 17), tf.float32) * 3
        mask_18 = tf.cast(tf.equal(y_true_class, 18), tf.float32) * 5  
        mask_20 = tf.cast(tf.equal(y_true_class, 20), tf.float32) * 7  

        mask_14 = tf.reshape(mask_14, tf.shape(mae_loss))
        mask_15 = tf.reshape(mask_15, tf.shape(mae_loss))
        mask_16 = tf.reshape(mask_16, tf.shape(mae_loss))
        mask_17 = tf.reshape(mask_17, tf.shape(mae_loss))
        mask_18 = tf.reshape(mask_18, tf.shape(mae_loss))
        mask_20 = tf.reshape(mask_20, tf.shape(mae_loss))
    
        multiplier = 1 + mask_14 + mask_15 + mask_16 + mask_17 + mask_20
        adjusted_mae_loss = mae_loss * multiplier
        
        #tf.print("y_true_class_ref:", y_true_class_refs, summarize=-1)
        #tf.print("weights:", weights, summarize=-1)
        #tf.print("y_true:", y_true, summarize=-1)
        #tf.print("y_true_class:", y_true_class, summarize=-1)
        #tf.print("y_true_reg:", y_true_reg, summarize=-1)
        #tf.print("y_pred:", y_pred, summarize=-1)
        #tf.print("mask:", mask)
        #tf.print("Mean Absolute Error:", mae_loss)
        #tf.print("Adjusted Mean Absolute Error:", adjusted_mae_loss)
        
        return tf.reduce_mean(adjusted_mae_loss)
    
    return weighted_loss

def evaluate(meta_test, y_pred_reg, y_pred_class, measureName, distortion_mapping, single, classify=False, name='results.txt'):
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
    print(f'Results saved to {filepath}')
    return lcc