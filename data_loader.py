import numpy as np
import pandas as pd
import cv2
import urllib.request
import subprocess
import os
from scipy.signal import convolve2d
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
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
        self.metadata = {'train': None, 'val': None, 'test': None}

        ### Attributes to be declared within the child class ###
        self.url = ''      # URL of the dataset (if applicable)
        self.exdir = ''    # Directory where the exctracted dataset is stored
        self.score = ''    # MOS/DMOS column name
        self.images = ''   # Directory where the images are stored
        self.archive_file = '' # Path to the rar/zip file
        
    def data_exist(self):
        '''Check if patch files are present in the directory.'''
        return (os.path.exists(os.path.join(self.exdir, 'X_train.npy')) and os.path.exists(os.path.join(self.exdir, 'y_train.npy')) and
                os.path.exists(os.path.join(self.exdir, 'X_val.npy')) and os.path.exists(os.path.join(self.exdir, 'y_val.npy')) and
                os.path.exists(os.path.join(self.exdir, 'X_test.npy')) and os.path.exists(os.path.join(self.exdir, 'y_test.npy')))

    def save_data(self, datasets):
        '''Save the data to disk.'''
        for name, data in datasets.items():
            (X, y) = data
            np.save(os.path.join(self.exdir, f'X_{name}.npy'), X)
            np.save(os.path.join(self.exdir, f'y_{name}.npy'), y)
        #logger.info(f"Data saved successfully.")
    
    def download(self, extract_in='databases'):
        '''Download the dataset from the URL and extract it to the directory.
        Args:
            extract_in (str, optional): Provide if the dataset is not extracted into a folder named after the file
        Note:
            You need to provide path into WinRAR or 7zip exe file.
        '''
        def extract_with_winrar(extract_in='databases'):
            try:
                logging.info(f"Extracting {self.archive_file} with WinRAR...")
                subprocess.run([self.winrar, 'x', self.archive_file, extract_in], capture_output=True, text=True)
                logging.info(f"Dataset extracted in '{self.exdir}'.")
                return True
            except Exception as e:
                logging.error(f"Error using WinRAR: {e}")
                return False

        def extract_with_7zip(self, extract_in='databases'):
            try:
                logging.info(f"Extracting {self.archive_file} with 7-Zip...")
                subprocess.run([self.sevenzip, 'x', '-aoa', self.archive_file, f'-o{extract_in}'], capture_output=True, text=True)
                logging.info(f"Dataset extracted in '{self.exdir}'.")
                return True
            except Exception as e:
                logging.error(f"Error using 7-Zip: {e}.")
                return False

        def extract_rar(self, extract_in='databases'):
            try:
                subprocess.run(['unrar', 'x', self.archive_file, extract_in], capture_output=True, text=True)
                logger.info(f"Dataset extracted in '{self.exdir}'.")
                return True
            except Exception as e:
                logger.error(f"Error using WinRAR: {e}")
                return False

        def extract_zip(self, extract_in='databases'):
            try:
                subprocess.run(['7z', 'x', self.archive_file, '-o' + extract_in], capture_output=True, text=True)
                logger.info(f"Dataset extracted in '{self.exdir}'.")
                return True
            except Exception as e:
                logger.error(f"Error using 7-Zip: {e}.")
                return False
            
        try:
            if os.path.exists(self.images or self.data_exist):
                logging.info("Dataset found.")
                return True
            if not os.path.exists(self.archive_file):
                logging.warning(f"Dataset not found. Downloading from {self.url}...")
                urllib.request.urlretrieve(self.url, self.archive_file, reporthook=self.track_download_progress)
            if os.name == 'posix':
                if archive_file.endswith('.rar'):
                    extract_rar(extract_in)
                elif archive_file.endswith('.zip'):
                    extract_zip(extract_in)
                else:
                    print(f"Unsupported file type: {self.archive_file}")
                    return False
            else:
                if not extract_with_winrar(extract_in):
                    extract_with_7zip(extract_in)
        except Exception as e:
            logging.error(f"Failed to download or extract dataset: {e}.")
            return False  
        return True
    
    def track_download_progress(self, count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f'\rDownloading: {percent}% ', end='\r')

    def split_data(self, data):
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=40)
        train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=40)
        return train_data, val_data, test_data

    def preprocess(self, datasets, patch_size=32):
        '''
        For datset in {trainig, validation, test}:
            Reads file paths to images from dataframes -> 
            Normalizes images and slices them to patches -> 
            Saves file path to patches to dataframe.
        Arg: Dictionary of dataframes
        Return: Dictionary of dataframes
        '''
        data = dict()
        tensors = dict()
        total_images = sum(len(data) for data in datasets.values())
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
                    X.append(patch)
                    y.append(score)
                    patch_count += 1
        
        for (name, dataset) in datasets.items():
            output_dir_full = os.path.join(self.exdir, 'normalized_distorted_images', name, 'full')
            output_dir_patches = os.path.join(self.exdir, 'normalized_distorted_images', name, 'patches')
            os.makedirs(output_dir_full, exist_ok=True)
            os.makedirs(output_dir_patches, exist_ok=True)
            patches = []
            X, y = [], []
            for row in dataset.itertuples(index=False):
                filename = row[0]
                extension = os.path.splitext(filename)[1]
                score = row[1]
                distortion = row[2]
                image_path = os.path.join(self.images, filename)
                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Failed to load image: {filename}")
                    continue
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_normalized = normalize_image(image_gray)
                filename = f'NORM_{filename.lower()}'
                cv2.imwrite(os.path.join(output_dir_full, filename), image_normalized)
                slice_image(image_normalized, patch_size)
                processed_images += 1  
                print(f"Preprocessed {processed_images}/{total_images} images.", end='\r') 

            patches = pd.DataFrame(patches, columns=['image', self.score, 'distortion'])
            patches.to_csv(os.path.join(self.exdir, f'{name}-metadata.csv'), index=False)
            data[name] = patches
            X = np.array(X)
            y = np.array(y)
            X = X[..., np.newaxis]
            tensors[name] = (X, y)
        logger.info('Mapping data to TensorFlow format...')
        return data, tensors

    def encode(self, dataframes):
        '''Encodes distortion labels into one-hot vectors.'''
        for i in range(len(dataframes)):
            dists = dataframes[i]['distortion']
            le = LabelEncoder()
            y_class_encoded = le.fit_transform(dists)
            dists_one_hot = to_categorical(y_class_encoded, num_classes=13).astype(int)
            dataframes[i]['distortion_encoded'] = [np.array(one_hot) for one_hot in dists_one_hot]
            dataframes[i] = dataframes[i].drop(['distortion'], axis=1)
        return dataframes

# TO DO:
# filter pristine images
# create config file
class tid2013_loader(database_loader):
    def __init__(self):
        super().__init__()
        self.url = 'https://www.ponomarenko.info/tid2013/tid2013.rar'
        self.exdir = os.path.join(self.catalogue, 'tid2013')
        self.score = 'MOS'
        self.images = os.path.join(self.exdir, 'distorted_images')
        self.archive_file = os.path.join(self.catalogue, 'tid2013.rar')
        self.distortion_mapping = {1: 'wn', 2:'wnc', 3:'scn', 4:'mn', 5:'hfn', 
                                   6:'in', 7:'qn', 8: 'gblur', 9:'idn', 10: 'jpeg', 
                                   11: 'jp2k', 12:'jpegte', 13:'jp2kte'} # According to TID2013 documentation

        os.makedirs(self.exdir, exist_ok=True)
        self.download(extract_in=self.exdir)
        
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

    def prepare_data(self, filter=True):
        data_path = os.path.join(self.exdir, 'mos_with_names.txt')
        data = pd.read_csv(data_path, header=None, delimiter=' ')
        data = data.iloc[:, [1, 0]]  # swap column order
        data.columns = ['image', 'MOS']
        data['distortion'] = data['image'].apply(lambda x: self.distortion_mapping.get(int(x.split('_')[1]), 'other'))
        if filter:
            data = data[data['distortion'].isin(self.distortion_mapping.values())]
        data.to_csv(os.path.join(self.exdir,'mos_with_names.csv'), index=False)

        datasets = {name: dataset for (name, dataset) in zip(self.metadata.keys(), self.split_data(data))}
        datasets = self.preprocess(datasets)
        return datasets
    
class kadid10k_loader(database_loader): 
    def __init__(self):
        super().__init__()
        self.url = 'https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip'
        self.exdir = os.path.join(self.catalogue, 'kadid10k')
        self.score = 'DMOS'
        self.images = os.path.join(self.exdir, 'images')
        self.archive_file = os.path.join(self.catalogue, 'kadid10k.zip')
        self.distortion_mapping = {1: 'gblur', 2: 'lblur', 3: 'mblur', 4: 'cdiff', 5: 'cshift', # According to KADID-10k documentation
                                   6: 'cquant', 7: 'csat1', 8: 'csat2', 9: 'jp2k', 10: 'jpeg',
                                   11: 'wniose1', 12: 'wniose2', 13: 'inoise', 14: 'mnoise', 15: 'denoise',
                                   16: 'bright', 17: 'dark', 18: 'meanshft', 19: 'jit', 20: 'patch', 
                                   21: 'pixel', 22: 'quant', 23: 'cblock', 24: 'sharp', 25: 'contrst'} 
        
        self.download()
        
        if self.data_exist():
            logger.info("Loading data...")
            self.metadata = {name: pd.read_csv(os.path.join(self.exdir, f'{name}-metadata.csv')) for name in self.metadata.keys()}
            self.train, self.val, self.test = [(np.load(os.path.join(self.exdir, f'X_{name}.npy')), np.load(os.path.join(self.exdir, f'y_{name}.npy'))) for name in self.metadata.keys()]
        else:
            self.metadata = self.prepare_data()
            logger.info('Mapping data to TensorFlow format...')
            self.train, self.val, self.test = self.map2tf(self.metadata)
            self.save_data({'train': self.train, 'val': self.val, 'test': self.test})
        #self.train, self.val, self.test = self.encode(data)

        logging.info("Data loaded successfully.")

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
        