from tensorflow.keras import layers, models


'''def build_model(n_neurons1=800, n_neurons2=800, dropout_rate=0):

    inputs = layers.Input(shape=(32, 32, 1))
    
    x = layers.Conv2D(50, (7, 7), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(n_neurons1, activation='relu')(x)
    x = layers.Dense(n_neurons2, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    regression_output = layers.Dense(1, activation='linear', name='regression_output')(x)
    model = models.Model(inputs=inputs, outputs=regression_output)
    return model
'''
def build_model(n_neurons1=8, n_neurons2=8, dropout_rate=0):

    inputs = layers.Input(shape=(32, 32, 1))
    
    x = layers.Conv2D(50, (7, 7), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(n_neurons1, activation='relu')(x)
    x = layers.Dense(n_neurons2, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    regression_output = layers.Dense(1, activation='linear', name='regression_output')(x)
    model = models.Model(inputs=inputs, outputs=regression_output)
    return model

def build_model_82(n_neruons=512, dropout_rate=0.5): 
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