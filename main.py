import tensorflow as tf
import keras
import optuna
import argparse
import datetime

from data_loader import tid2013_loader, kadid10k_loader
from architectures import *
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K

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

### Training    Tuning     Loading the model ###
def learn(model, X_train, y_train, val, epochs, early_stopping, loss_function, batch_size, learning_rate):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss_function)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=val, callbacks=[early_stopping])
    del X_train, y_train, val
    # After training
    K.clear_session()
    y_pred_reg, y_pred_class = model.predict(X_test, verbose=1), None
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    modelname = f'model_{timestamp}.h5'
    model.save(modelname)
    lcc = evaluate(meta_test, y_pred_reg, y_pred_class, measureName, distortion_mapping, classify)
    return lcc

def tune(n_trials, X_train, y_train_reg, y_train_class, validation_data, epochs, classify):
    def objective(trial):
        n_neurons1 = trial.suggest_int('n_neurons1', 500, 1500)
        eta = trial.suggest_float('eta', 1e-3, 1e-2)
        dropout_rate1 =trial.suggest_float('dropout_rate1', 0, 0.8)
        batch_size = trial.suggest_int('batch_size', 20, 50)
        early_stopping = EarlyStopping(monitor='val_loss' if not classify else 'val_regression_output_loss', 
            patience=5, restore_best_weights=True)
        
        if classify:
            model = build_model(num_classes)
            loss = ['mean_absolute_error', 'sparse_categorical_crossentropy']
            loss_weights = [1.0, 0.2]
            y_train = [y_train_reg, y_train_class]
        else:
            model = build_model_82(n_neurons1, dropout_rate1)
            loss = 'mean_absolute_error'
            y_train = y_train_reg
    
        lcc = learn(model, X_train, y_train, validation_data, epochs, early_stopping, loss, batch_size, learning_rate=eta)
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
    parser.add_argument('command', choices=['learn', 'load', 'tune'], default = 'tune', help='What to do.')
    parser.add_argument('--training', choices=['tid2013', 'kadid10k'], type=str, default='tid2013', help='Database for training set (default: tid2013)')
    parser.add_argument('--test', choices=['tid2013', 'kadid10k'], type=str, help='Database for test set. If not specified - evaluation is done on the trainig set')
    parser.add_argument('--epochs', type=int, default = 1, help='Number of epochs (default: 1)')
    parser.add_argument('--n_trials', type=int, default = 1, help='Number of trials for hyperparameter tuning (default: 1)')
    parser.add_argument('--classify', action='store_true', help='Enable distortion classification')
    parser.add_argument('--use_val', action='store_true', help='Use validation set for training')
    args = parser.parse_args()

    if args.test == None:
        args.test = args.training
    training, test, epochs, classify, n_trials, use_val = args.training, args.test, args.epochs, args.classify, args.n_trials, args.use_val
    if args.command == 'tune':
        use_val = True

    ### Set-up for training ###
    databases = {'tid2013': tid2013_loader, 'kadid10k': kadid10k_loader}
    training_loader = databases[training](test, as_training=True)
    num_classes = training_loader.num_classes
    training_data = training_loader.metadata
    X = training_loader.X
    y_reg = training_loader.y_reg
    y_class = training_loader.y_class

    # Categorize quality scores
    #k = training_loader.quality_clusters
    #temp_x = training_data[training_loader.measureName].values.reshape(-1, 1) # rename this variable 
    #kmeans = KMeans(n_clusters=k, random_state=42)
    #kmeans.fit(temp_x)
    #training_data['class'] = kmeans.labels_
    #cluster_means = training_data.groupby('class')[training_loader.measureName].mean().reset_index()
    #ordered_clusters = cluster_means.sort_values(by=training_loader.measureName)
    #cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(ordered_clusters['class'])}
    #training_data['class']= training_data['class'].map(cluster_mapping)
    
    # Divide data
    if test==training:
        measureName = training_loader.measureName
        distortion_mapping = training_loader.distortion_mapping
        meta_train, meta_val, meta_test  = split_data(training_data, measureName)
        train_indices, val_indices, test_indices = meta_train.index, meta_val.index, meta_test.index
        X_test = X[test_indices]
        y_test_reg = y_reg[test_indices]
        y_test_class = y_class[test_indices]     

    else:
        test_loader = databases[test](training)
        test_data = test_loader.metadata
        test_measureName = test_loader.measureName
        training_measureName = training_loader.measureName
        measureName = test_measureName
        #distortion_mapping = getattr(training_loader, f'distortion_mapping_{test}')
        distortion_mapping = test_loader.distortion_mapping
        training_data[training_measureName] = mos2dmos(training_data[training_measureName], test_data[test_measureName])
        meta_train, meta_val = split_data(training_data, training_measureName, validation=False)
        meta_test = test_loader.metadata
        train_indices, val_indices = meta_train.index, meta_val.index    
        X_test =  test_loader.X
        y_test_reg =  test_loader.y_reg
        y_test_class =  test_loader.y_class
    del meta_train, meta_val
    # Map indices to data       
    X_train, X_val = X[train_indices], X[val_indices] 
    y_train_reg, y_val_reg = y_reg[train_indices], y_reg[val_indices]
    y_train_class, y_val_class = y_class[train_indices], y_class[val_indices]

    if not use_val:
        X_train = np.concatenate((X_train, X_val), axis=0)
        y_train_reg = np.concatenate((y_train_reg, y_val_reg), axis=0)
        y_train_class = np.concatenate((y_train_class, y_val_class), axis=0)
        early_stopping = EarlyStopping(monitor='loss' if not classify else 'regression_output_loss', 
            patience=5, restore_best_weights=True)
        validation_data = None 
    else:
        validation_data = (X_val, y_val_reg)
        early_stopping = EarlyStopping(monitor='val_loss' if not classify else 'val_regression_output_loss', 
            patience=10, restore_best_weights=True)

    if args.command == 'learn':
        if classify:
            model = build_model(num_classes)
            loss = ['mean_absolute_error', 'sparse_categorical_crossentropy']
            loss_weights = [1.0, 0.2]
            batch_size = 32
            y_train = [y_train_reg, y_train_class]
        else:
            model = build_model_82()
            loss = 'mean_absolute_error'
            batch_size = 22
            y_train = y_train_reg
        learn(model, X_train, y_train, validation_data, epochs, early_stopping, 
              loss, batch_size, learning_rate=0.001)

    elif args.command == 'tune':
        tune(n_trials, X_train, y_train_reg, y_train_class, validation_data, epochs, classify)

                
    elif args.command == 'load':
        model = models.load_model('model_10-16_15-51.h5')
        early_stopping = EarlyStopping(monitor='loss' if not classify else 'regression_output_loss', 
            patience=5, restore_best_weights=True)
        validation_data = None
        if classify:
            y_pred_reg, y_pred_class = model.predict(X_test, verbose=2)
        else:
            y_pred_class = None
            #model.fit(X_train, y_train_reg, epochs=epochs, batch_size=22, verbose=2, validation_data=validation_data, 
            #    callbacks=[early_stopping])
            y_pred_reg, y_pred_class = model.predict(X_test, verbose=1), None
        evaluate(meta_test, y_pred_reg, y_pred_class, measureName, distortion_mapping, classify)
        model.save('iqa.h5')
