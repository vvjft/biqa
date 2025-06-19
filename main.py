import tensorflow as tf
import numpy as np
import keras
import optuna
import argparse
import datetime
import gc
import copy
import random

from data_loader import tid2013_loader, kadid10k_loader
from architectures import *
from utils import split_data, mos2dmos, evaluate
from tensorflow.keras.callbacks import EarlyStopping

### Training    Tuning     Loading the model ###
def train(model, X_train, y_train, val, epochs, early_stopping, loss_function, batch_size, learning_rate):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss_function)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=val, callbacks=[early_stopping]) 
    y_pred_reg = model.predict(X_test, verbose=1)


    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    modelname = f'model_{timestamp}.h5'
    model.save(modelname)
    #data = copy.deepcopy(meta_test)

    #lcc = evaluate(data, y_pred_reg, measureName, distortion_mapping)
    #del data
    return y_pred_reg

def tune(n_trials, X_train, y_train_reg, validation_data, epochs):
    def objective(trial):
        #n_neurons1 = trial.suggest_int('n_neurons1', 500, 1000)
        #n_neurons2 = trial.suggest_int('n_neurons2', 500, 1000)
        #eta = trial.suggest_float('eta', 1e-3, 1e-2)
        n_neurons1 = trial.suggest_int('n_neurons1', 800,2000)
        #n_neurons2 = trial.suggest_int('n_neurons2', 800,2000)
        eta = trial.suggest_float('eta', 1e-3, 1e-1)
        dropout_rate1 = trial.suggest_float('dropout_rate1', 0, 0.7)
        #batch_size = trial.suggest_int('batch_size', 20, 50)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001, baseline=0.90)
        
        model = build_model_82(n_neurons1, dropout_rate=dropout_rate1)
        loss = 'mean_absolute_error'
        y_train = y_train_reg
        
        y_pred_reg = train(model, X_train, y_train, validation_data, epochs, early_stopping, loss_function=loss, batch_size=22, learning_rate=eta)
        lcc = evaluate(meta_test, y_pred_reg, measureName, distortion_mapping)
        
        tf.keras.backend.clear_session()
        gc.collect()
        return lcc
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    parser = argparse.ArgumentParser(description='Description to be filled')
    parser.add_argument('command', choices=['train', 'load', 'tune'], default = 'tune', help='What to do.')
    parser.add_argument('--training_set', choices=['tid2013', 'kadid10k'], type=str, default='tid2013', help='Database for training set (default: tid2013)')
    parser.add_argument('--test_set', choices=['tid2013', 'kadid10k'], type=str, help='Database for test set. If not specified - evaluation is done on the trainig set')
    parser.add_argument('--epochs', type=int, default = 1, help='Number of epochs (default: 1)')
    parser.add_argument('--n_trials', type=int, default = 1, help='Number of trials for hyperparameter tuning (default: 1)')
    parser.add_argument('--use_val', action='store_true', help='Use validation set for training')
    args = parser.parse_args()

    if args.test_set == None:
        args.test_set = args.training_set
    training_set, test_set, epochs, n_trials, use_val = args.training_set, args.test_set, args.epochs, args.n_trials, args.use_val
    if args.command == 'tune':
        use_val = True
    
    ### Set-up for training ###
    databases = {'tid2013': tid2013_loader, 'kadid10k': kadid10k_loader}
    training_loader = databases[training_set]()
    num_classes = training_loader.num_classes
    training_data = training_loader.metadata
    X = training_loader.X
    y_reg = training_loader.y_reg

    # Split data
    if test_set == training_set:
        measureName = training_loader.measureName
        distortion_mapping = training_loader.distortion_mapping
        meta_train, meta_val, meta_test  = split_data(training_data, measureName)
        train_indices, val_indices, test_indices = meta_train.index, meta_val.index, meta_test.index
        X_test = X[test_indices]
        y_test_reg = y_reg[test_indices]   
    else:
        test_loader = databases[test_set](training_set)
        test_data = test_loader.metadata
        test_measureName = test_loader.measureName
        training_measureName = training_loader.measureName
        measureName = test_measureName
        distortion_mapping = test_loader.distortion_mapping
        training_data[training_measureName] = mos2dmos(training_data[training_measureName], test_data[test_measureName])
        meta_train, meta_val = split_data(training_data, training_measureName, validation=False)
        meta_test = test_loader.metadata
        train_indices, val_indices = meta_train.index, meta_val.index    
        X_test =  test_loader.X
        y_test_reg =  test_loader.y_reg
        y_test_class =  test_loader.y_class
    
    X_train, X_val = X[train_indices], X[val_indices] 
    y_train_reg, y_val_reg = y_reg[train_indices], y_reg[val_indices]

    if not use_val:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train_reg = np.concatenate((y_train_reg, y_val_reg), axis=0)
        early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        validation_data = None 
    else:
        validation_data = (X_val, y_val_reg)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    if args.command == 'train':   
        model = build_model_82()
        loss = 'mse'
        batch_size = 22
        y_train = y_train_reg
        y_pred_reg = train(model, X_train, y_train, validation_data, epochs, early_stopping, 
              loss, batch_size, learning_rate=0.001)
        lcc = evaluate(meta_test, y_pred_reg, measureName, distortion_mapping)

    elif args.command == 'tune':
        tune(n_trials, X_train, y_train_reg, validation_data, epochs)
              
    elif args.command == 'load':
        model = models.load_model('85.h5')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        #validation_data = None
        #model.fit(X_train, y_train_reg, epochs=epochs, batch_size=22, verbose=1, validation_data=validation_data, 
        #    callbacks=[early_stopping])
        y_pred_reg = model.predict(X_test, verbose=1)
        evaluate(meta_test, y_pred_reg, measureName, distortion_mapping)
        model.summary()
        model.save('iqa.h5')
