import os
from tensorflow.keras import layers, models
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
 
if os.name == 'posix':
    from data_loader_linux import tid2013_loader, kadid10k_loader
else:
    from data_loader import tid2013_loader, kadid10k_loader

def build_model():
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

def group_results(metadata, results):
    test_table = metadata['test'] 
    test_table['pred_MOS'] = results
    test_table['prefix'] = test_table['image'].str.extract(r'(^.*_i\d+_\d+_\d+)_patch')

    grouped = test_table.groupby('prefix').agg(
        MOS=('MOS', 'first'),  
        averaged_pred_MOS=('pred_MOS', 'mean'),
        distortion=('distortion', 'first')  
    ).reset_index()

    grouped.rename(columns={'prefix': 'image'}, inplace=True)
    grouped.to_csv('grouped.csv')
    return grouped

def calculate_metrics(grouped):
    correlation, _ = pearsonr(grouped['MOS'], grouped['averaged_pred_MOS'])
    print("LCC:", correlation)

    mae = mean_absolute_error(grouped['MOS'], grouped['averaged_pred_MOS'])
    print("mae:", mae)

def main():
    data_loader = tid2013_loader()
    metadata = data_loader.metadata
    X_train, y_train = data_loader.train
    X_val, y_val = data_loader.val
    X_test, y_test = data_loader.test
    model = build_model()
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(X_train, y_train, epochs=2, validation_data=(X_val, y_val), verbose=2)
    y_pred = model.predict(X_test, verbose=2)

    grouped = group_results(metadata, y_pred)
    calculate_metrics(grouped)

if __name__ == "__main__":
    main()

