import numpy as np
import pandas as pd
import cv2
import urllib.request
import subprocess
import os
from scipy.signal import convolve2d
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging

os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file_path = os.path.join('logs',f'warnings_{timestamp}.log')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear() 
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.WARNING)  
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

class database_loader:
    '''Parent class for database-specific loaders.'''
    def __init__(self):
        self.catalogue = 'databases'
        self.winrar = 'D:\\Programy\\WinRAR\\WinRAR.exe'
        self.sevenzip = 'D:\\Programy\\7-Zip\\7z.exe'

        ### Attributes to be declared within the child class ###
        self.url = ''      # URL of the dataset (if applicable)
        self.exdir = ''    # Directory where the exctracted dataset is stored
        self.measureName = ''    # MOS/DMOS column name
        self.images_dir = ''   # Directory where the images are stored
        self.archive_file = '' # Path to the rar/zip file
        self.metadata = None
        
    def data_exist(self):
        '''Check if patch files are present in the directory.'''
        return os.path.exists(os.path.join(self.exdir, 'metadata.csv')) and os.path.exists(os.path.join(self.exdir, 'X.npz'))
    
    def download(self, extract_in='databases'):
        '''Download the dataset from the URL and extract it to the directory.
        Args:
            extract_in (str, optional): Provide if the dataset is not extracted into a folder named after the file
        Note:
            You need to provide path into WinRAR or 7zip exe file.
        '''
        def track_download_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f'\rDownloading: {percent}% ', end='\r')

        def extract_in_posix(extract_in='databases'):
            try:
                logging.info(f"Extracting {self.archive_file}...")
                if self.archive_file.endswith('.rar'):    
                    subprocess.run(['unrar', 'x', self.archive_file, extract_in], capture_output=True, text=True)
                else:
                    subprocess.run(['7z', 'x', self.archive_file, '-o' + extract_in], capture_output=True, text=True)
            except Exception as e:
                logger.error(f"Error while exctracting: {e}")
                return False
            else:
                logging.info(f"Dataset extracted in '{self.exdir}'.")
                return True

        def extract_in_windows(extract_in='databases'):
            try:
                if self.archive_file.endswith('.rar'):
                    logging.info(f"Extracting {self.archive_file} with WinRAR...")
                    subprocess.run([self.winrar, 'x', self.archive_file, extract_in], capture_output=True, text=True)
                else:
                    logging.info(f"Extracting {self.archive_file} with 7-Zip...")
                    subprocess.run([self.sevenzip, 'x', '-aoa', self.archive_file, f'-o{extract_in}'], capture_output=True, text=True)
            except Exception as e:
                logger.error(f"Error while exctracting: {e}")
                return False
            else:
                logging.info(f"Dataset extracted in '{self.exdir}'.")
                return True

        try:
            if os.path.exists(self.images_dir or self.data_exist):
                logging.info("Dataset found.")
                return True
            if not os.path.exists(self.archive_file):
                logging.warning(f"Dataset not found. Downloading from {self.url}...")
                urllib.request.urlretrieve(self.url, self.archive_file, reporthook=track_download_progress)
            if os.name == 'posix':
                extract_in_posix(extract_in)
            else:
                extract_in_windows(extract_in)
        except Exception as e:
            logging.error(f"Failed to download or extract dataset: {e}.")
            return False 
        else: 
            return True
    
    def preprocess(self, data, patch_size=32):
        total_images = len(data)
        processed_images = 0
        logging.info('Preprocessing images...')

        def normalize_image(patch, P=3, Q=3, C=1):
            kernel = np.ones((P, Q)) / (P * Q)
            patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
            patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
            patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
            patch_ln = (patch - patch_mean) / patch_std
            return patch_ln.astype('float32')
        
        def slice_image(image, patch_size=32):
            height, width = image.shape[:2]
            num_patches_y = height // patch_size
            num_patches_x = width // patch_size
            patch_count = 0
            for i in range(num_patches_y):
                for j in range(num_patches_x):
                    patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                    patch_path = os.path.join(output_dir_patches, f"{os.path.splitext(filename)[0]}_patch_{patch_count}{extension}")
                    patch_filename = f"{os.path.splitext(filename)[0]}_patch_{patch_count}{extension}"     
                    cv2.imwrite(patch_path, patch)
                    patches.append([patch_filename, score, distortion])
                    X[patch_filename] = patch
                    patch_count += 1
        
        output_dir_patches = os.path.join(self.exdir, 'patches')
        os.makedirs(output_dir_patches, exist_ok=True)
        patches = []
        X = {}
        for idx, row in data.iterrows():
            filename = row['image']
            score = row[self.measureName]
            distortion = row['distortion']

            extension = os.path.splitext(filename)[1]
            image_path = os.path.join(self.images_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                logging.warning(f'Failed to load image: {filename}')
            else:
                image_normalized = normalize_image(image)
                filename = f'NORM_{filename.lower()}'
                slice_image(image_normalized, patch_size)
                processed_images += 1  
                print(f'Preprocessed {processed_images}/{total_images} images.', end='\r') 
                
        patches = pd.DataFrame(patches, columns = ['image', self.measureName, 'distortion'])
        patches.to_csv(os.path.join(self.exdir, 'metadata.csv'), index=False)
        np.savez(os.path.join(self.exdir, 'X.npz'), **X)
                  
        return patches, X
        
    def split_data(self, metadata):
        logger.info('Splitting data...')
        metadata['image_id'] = metadata['image'].apply(lambda x: '_'.join(x.split('_')[:4]))
        groups = metadata.groupby('image_id').agg(list).reset_index()
        train, test = train_test_split(groups, test_size=0.2, random_state=40)
        train, val = train_test_split(train, test_size=0.25, random_state=40)
        
        self.meta_train = train.explode(['image', self.measureName, 'distortion'])
        self.meta_val = val.explode(['image', self.measureName, 'distortion'])
        self.meta_test = test.explode(['image', self.measureName, 'distortion'])
        
        self.meta_train.to_csv(os.path.join(self.exdir, 'meta_train.csv'))
        self.meta_val.to_csv(os.path.join(self.exdir, 'meta_val.csv'))
        self.meta_test.to_csv(os.path.join(self.exdir, 'meta_test.csv'))
        
        
    def map2tf(self):
        logger.info('Mapping data to TensorFlow format...')
        self.y_train_reg, self.y_val_reg, self.y_test_reg = np.array(self.meta_train[self.measureName], dtype=np.float32), np.array(self.meta_val[self.measureName], dtype=np.float32), np.array(self.meta_test[self.measureName], dtype=np.float32)
        self.y_train_class, self.y_val_class, self.y_test_class = np.array(self.meta_train['distortion'], dtype=np.int64), np.array(self.meta_val['distortion'], dtype=np.int64), np.array(self.meta_test['distortion'], dtype=np.int64)
        #self.X_train, self.X_valid, self.X_test = np.stack(self.meta_train['X'].values), np.stack(self.meta_val['X'].values), np.stack(self.meta_test['X'].values) 
        self.X_train = np.array([self.X[image] for image in self.meta_train['image'] if image in self.X])[..., np.newaxis]
        self.X_val = np.array([self.X[image] for image in self.meta_val['image'] if image in self.X])[..., np.newaxis]
        self.X_test = np.array([self.X[image] for image in self.meta_test['image'] if image in self.X])[..., np.newaxis]
        '''
        np.save(os.path.join(self.exdir, y_train_reg), 'y_train_reg.npy')
        np.save(os.path.join(self.exdir, y_val_reg), 'y_val_reg.npy')
        np.save(os.path.join(self.exdir, y_test_reg), 'y_test_reg.npy')
        np.save(os.path.join(self.exdir, y_train_class), 'y_train_class.npy')
        np.save(os.path.join(self.exdir, y_val_class), 'y_test_class.npy')
        np.save(os.path.join(self.exdir, y_test_class), 'y_test_class.npy')
        
        np.savez(os.path.join(self.exdir, 'X_train.npz'), X_train)
        np.savez(os.path.join(self.exdir, 'X_val.npz'), X_val)
        np.savez(os.path.join(self.exdir, 'X_test.npz'), X_test)
        '''
        


# TO DO:
# filter pristine images
# create config file
class tid2013_loader(database_loader):
    def __init__(self):
        super().__init__()
        self.url = 'https://www.ponomarenko.info/tid2013/tid2013.rar'
        self.exdir = os.path.join(self.catalogue, 'tid2013')
        self.measureName = 'MOS'
        self.images_dir = os.path.join(self.exdir, 'distorted_images')
        self.archive_file = os.path.join(self.catalogue, 'tid2013.rar')
        self.distortion_mapping = {1: 'wn', 2:'wnc', 3:'scn'}#, 4:'mn', 5:'hfn', 
                                   #6:'in', 7:'qn', 8: 'gblur', 9:'idn', 10: 'jpeg', 
                                   #11: 'jp2k', 12:'jpegte', 13:'jp2kte'} # According to TID2013 documentation

        os.makedirs(self.exdir, exist_ok=True)
    
        if self.download(extract_in = self.exdir):
            if self.data_exist():
                logger.info("Loading data...")
                self.metadata = pd.read_csv(os.path.join(self.exdir, 'metadata.csv'))
                self.X = np.load(os.path.join(self.exdir, 'X.npz'))
                self.split_data(self.metadata)
                self.map2tf()
            else:
                self.metadata, self.X = self.prepare_data()
                self.split_data(self.metadata)
                self.map2tf()
            
            logging.info("Data loaded successfully.")
        else:
            logging.error("Cannot download or extract database.")
            
    def prepare_data(self, filter = True):
        data_path = os.path.join(self.exdir, 'mos_with_names.txt')
        data = pd.read_csv(data_path, header=None, delimiter=' ')
        data = data.iloc[:, [1, 0]]  # swap column order
        data.columns = ['image', 'MOS']
        data['distortion'] = [int(img.split('_')[1]) for img in data['image']]
        if filter:
            data = data[data['distortion'].isin(self.distortion_mapping.keys())]
        data = self.preprocess(data)
        return data
    
class kadid10k_loader(database_loader): 
    def __init__(self):
        super().__init__()
        self.url = 'https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip'
        self.exdir = os.path.join(self.catalogue, 'kadid10k')
        self.measureName = 'DMOS'
        self.images_dir = os.path.join(self.exdir, 'images')
        self.archive_file = os.path.join(self.catalogue, 'kadid10k.zip')
        self.distortion_mapping = {1: 'gblur', 2: 'lblur', 3: 'mblur', 4: 'cdiff', 5: 'cshift', # According to KADID-10k documentation
                                   6: 'cquant', 7: 'csat1', 8: 'csat2', 9: 'jp2k', 10: 'jpeg',
                                   11: 'wniose1', 12: 'wniose2', 13: 'inoise', 14: 'mnoise', 15: 'denoise',
                                   16: 'bright', 17: 'dark', 18: 'meanshft', 19: 'jit', 20: 'patch', 
                                   21: 'pixel', 22: 'quant', 23: 'cblock', 24: 'sharp', 25: 'contrst'} 
        
        if self.download():
        
            if self.data_exist():
                logger.info("Loading data...")
                self.metadata = {name: pd.read_csv(os.path.join(self.exdir, f'{name}-metadata.csv')) for name in self.metadata.keys()}
                self.train, self.val, self.test = [(np.load(os.path.join(self.exdir, f'X_{name}.npy')), np.load(os.path.join(self.exdir, f'y_{name}.npy'))) for name in self.metadata.keys()]
            else:
                self.metadata, tensors = self.prepare_data()
                self.train, self.val, self.test = tensors.values()
                self.save_data({'train': self.train, 'val': self.val, 'test': self.test})
            #self.train, self.val, self.test = self.encode(data)

            logging.info("Data loaded successfully.")
        else:
            logging.error("Cannot download or extract database.")

    def prepare_data(self, filter=True):
        data_path = os.path.join(self.exdir, 'dmos.csv')
        data = pd.read_csv(data_path, header=0, usecols=[0, 2])
        data.columns = ['image', 'DMOS']
        data['distortion'] = data['image'].apply(lambda x: self.distortion_mapping.get(int(x.split('_')[1]), 'other'))
        #if True:
            #data = data[data['distortion'].isin(self.distortion_mapping.values())]
        data.to_csv(os.path.join(self.exdir,'dmos_with_names.csv'), index=False)

        datasets = {name: dataset for (name, dataset) in zip(self.metadata.keys(), self.split_data(data))}
        datasets = self.preprocess(datasets)
        return datasets
        


from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

  


def group_results(metadata, measureName):

    metadata['prefix'] = metadata['image'].str.extract(r'(^.*_i\d+_\d+_\d+)_patch')

    grouped = metadata.groupby('prefix').agg(
        measure=(measureName, 'first'),  
        pred_measure=('pred_MOS', 'mean'),
        distortion=('distortion', 'first')  
    ).reset_index()

    grouped.rename(columns={'prefix': 'image', 'measure': measureName, 'pred_measure': f'pred_{measureName}'}, inplace=True)
    print(grouped.head())
    grouped.to_csv('grouped.csv', index=False, header=True)
    return grouped

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

num_classes = 13
data_loader = tid2013_loader()
meta_test = data_loader.meta_test
X_train, y_train_reg, y_train_class = data_loader.X_train, data_loader.y_train_reg, data_loader.y_train_class 
X_val, y_val_reg, y_val_class = data_loader.X_val, data_loader.y_val_reg, data_loader.y_val_class
X_test, y_test_reg, y_test_class = data_loader.X_test, data_loader.y_test_reg, data_loader.y_test_class



model = build_model(num_classes)
model.compile(optimizer='adam', loss=['mae', 'sparse_categorical_crossentropy'])
model.fit(X_train, [y_train_reg,  y_train_class], epochs=5, batch_size=32, verbose=2)

y_pred_reg, y_pred_class = model.predict(X_test, verbose=2)


meta_test['MOS'] = pd.to_numeric(meta_test['MOS'], errors='coerce').astype('float32')
meta_test['distortion'] = pd.to_numeric(meta_test['distortion'], errors='coerce').astype('int64')
indices = np.argmax(y_pred_class, axis=1)
meta_test['pred_MOS'] = y_pred_reg.flatten()
meta_test['pred_distortion'] = indices
meta_test.to_csv('meta_test.csv')
print(meta_test.head())
print(meta_test.info())
missing_values = meta_test.isnull().sum()
print(missing_values)
print(accuracy_score(meta_test['distortion'], meta_test['pred_distortion']))
