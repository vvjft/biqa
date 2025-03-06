from tensorflow.keras import layers, models


def build_model(n_neurons1=800, n_neurons2=800, dropout_rate=0):

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

