import numpy as np
import pandas as pd
import cv2
import urllib.request
import subprocess
import os
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

class database_loader:
    '''Parent class for database-specific loaders.'''
    def __init__(self):
        self.catalogue = 'databases'
        self.winrar_path = 'D:\\Programy\\WinRAR\\WinRAR.exe'
        self.sevenzip_path = 'D:\\Programy\\7-Zip\\7z.exe'
        ### Attributes to be declared within the child class ###
        self.url = ''      # URL of the dataset
        self.dir = ''      # Directory where the exctracted dataset is stored
        self.images = ''   # Directory where the images are stored
        self.rar_file = '' # Path to the rar file
        self.zip_file = '' # Path to the zip file

    def download(self, file, extract=True, extract_in='databases'):
        '''Download the dataset from the URL and extract it to the directory.
        Args:
            file: (str) either rar_file or zip_file
            extract: (bool) if True, extract the dataset (TO DO)
            extract_in (str, optional): Provide if the dataset is not extracted into a folder named after the file
        Note:
            The method uses WinRAR to extract the dataset. If WinRAR is not available, it uses 7-Zip.
        '''
        try:
            logging.info(f"Downloading dataset from {self.url}...")
            urllib.request.urlretrieve(self.url, file)

        except Exception as e:
            logging.error(f"Failed to download dataset: {e}.")
            return False  

        if not self.extract_with_winrar(file):
            self.extract_with_7zip(file)

    def extract_with_winrar(self, file, extract_in='databases'):
        try:
            logging.info(f"Extracting dataset using WinRAR...")
            subprocess.run([self.winrar_path, 'x', file, extract_in], capture_output=True, text=True)
            logging.info(f"Dataset extracted in '{self.dir}'.")
            return True
        except Exception as e:
            logging.error(f"Error using WinRAR: {e}")
            return False

    def extract_with_7zip(self, file, extract_in='databases'):
        try:
            logging.info(f"Extracting dataset using 7-Zip...")
            subprocess.run([self.sevenzip_path, 'x', '-aoa', file, f'-o{extract_in}'], capture_output=True, text=True)
            logging.info(f"Dataset extracted in '{self.dir}'.")
            return True
        except Exception as e:
            logging.error(f"Error using 7-Zip: {e}.")
            return False
        
    def save_data(self, data, set_type):
        '''Save the data to disk.'''
        X, y = data
        np.save(os.path.join(self.dir, f'X_{set_type}.npy'), X)
        np.save(os.path.join(self.dir, f'y_{set_type}.npy'), y)
        logging.info(f"{set_type} data saved successfully.")

    def load_data(self, set_type):
        '''Load the data from disk.'''
        X = np.load(os.path.join(self.dir, f'X_{set_type}.npy'))
        y = np.load(os.path.join(self.dir, f'y_{set_type}.npy'))
        #logging.info(f"{set_type} data loaded successfully.")
        return X, y

    def data_exist(self):
        '''Check if patch files are present in the directory.'''
        return (os.path.exists(os.path.join(self.dir, 'X_train.npy')) and os.path.exists(os.path.join(self.dir, 'y_train.npy')) and
                os.path.exists(os.path.join(self.dir, 'X_val.npy')) and os.path.exists(os.path.join(self.dir, 'y_val.npy')) and
                os.path.exists(os.path.join(self.dir, 'X_test.npy')) and os.path.exists(os.path.join(self.dir, 'y_test.npy')))

    def split_data(self, data):
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=40)
        train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=40)
        return train_data, val_data, test_data

    def preprocess(self, train_data, val_data, test_data, patch_size=32):

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
                    patch_path = os.path.join(output_dir_patches, f"{os.path.splitext(filename)[0]}_patch_{patch_count}.bmp")
                    patch_filename = f"{os.path.splitext(filename)[0]}_patch_{patch_count}.bmp"
                    cv2.imwrite(patch_path, patch)
                    self.patches.append([patch_filename, mos, distortion])
                    patch_count += 1

        sets = [(train_data, 'training'), (val_data, 'validation'), (test_data, 'test')]
        dfs = []
        total_images = sum(len(data) for data, _ in sets)
        processed_images = 0
        logging.info('Preprocessing images...')

        for (data, name) in sets:
            output_dir_full = os.path.join(self.dir, 'normalized_distorted_images', name, 'full')
            output_dir_patches = os.path.join(self.dir, 'normalized_distorted_images', name, 'patches')
            os.makedirs(output_dir_full, exist_ok=True)
            os.makedirs(output_dir_patches, exist_ok=True)
            self.patches = []
            for row in data.itertuples(index=False):
                filename = row[0]
                mos = row[1]
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
                print(f"Processed {processed_images}/{total_images} images.", end='\r') 

            patches = pd.DataFrame(self.patches, columns=['image', 'MOS', 'distortion'])
            dfs.append(patches)
        return dfs

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

    def map2tf(self, set_type, data):
        '''
        Maps data into format excected by TensorFlow: adds chanel dimension and stores data in arrays.
        set_type: 'training', 'validation', 'test' 
        data: DataFrame with columns: 'image', 'MOS', 'distortion' 
        '''
        images_dir = os.path.join(self.dir, 'normalized_distorted_images', set_type, 'patches')
        X, y = [], []
        for row in data.itertuples(index=False):
            filename = row[0]
            score = row[1]
            file_path = os.path.join(images_dir, filename)
            if filename.endswith(('.bmp', '.png')) and os.path.exists(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                X.append(img)
                y.append(score)
            else:
                logging.warning(f"File not found: {file_path}")
        X = np.array(X)
        y = np.array(y)
        X = X[..., np.newaxis]
        return X, y
# TO DO:
# error handling: when download=False, download anyway if dataset doesn't exist 
# error handling: extract when dataset is downloaded but not extracted
# add error handling in save() and load() methods
# save patches in format according to original images
class tid2013_loader(database_loader):
    def __init__(self, download=True):
        super().__init__()
        self.url = 'https://www.ponomarenko.info/tid2013/tid2013.rar'
        self.dir = os.path.join(self.catalogue, 'tid2013')
        self.images = os.path.join(self.dir, 'distorted_images')
        self.rar_file = os.path.join(self.catalogue, 'tid2013.rar')
        self.distortion_mapping = {1: 'wn', 2:'wnc', 3:'scn', 4:'mn', 5:'hfn', 
                                   6:'in', 7:'qn', 8: 'gblur', 9:'idn', 10: 'jpeg', 
                                   11: 'jp2k', 12:'jpegte', 13:'jp2kte'} # According to TID2013 documentation

        os.makedirs(self.dir, exist_ok=True)
        if download:
            self.download(self.url, self.rar_file, extract_in=self.dir)
        elif not os.path.exists(self.rar_file):
            logging.info("Dataset not found. Downloading...")
            self.download(self.url, self.rar_file)
        
        if self.data_exist():
            logging.info("Patch files found. Loading patched data...")
            self.train = self.load_data('train')
            self.val = self.load_data('val')
            self.test= self.load_data('test')
        else:
            data = self.prepare_data()
            logging.info('Mapping data to TensorFlow format...')
            self.train = self.map2tf('training', data[0])
            self.val = self.map2tf('validation', data[1])
            self.test = self.map2tf('test', data[2])
            self.save_data(self.train, 'train')
            self.save_data(self.val, 'val')
            self.save_data(self.test, 'test')
        #self.train, self.val, self.test = self.encode(data)

        logging.info("Data loaded successfully.")

    def prepare_data(self, filter=True):
        data_path = os.path.join(self.dir, 'mos_with_names.txt')
        data = pd.read_csv(data_path, header=None, delimiter=' ')
        data = data.iloc[:, [1, 0]]  # swap column order
        data.columns = ['image', 'MOS']
        data['distortion'] = data['image'].apply(lambda x: self.distortion_mapping.get(int(x.split('_')[1]), 'other'))
        if filter:
            data = data[data['distortion'].isin(self.distortion_mapping.values())]
        data.to_csv(os.path.join(self.dir,'mos_with_names.csv'), index=False)

        train_data, val_data, test_data = self.split_data(data)
        train_data, val_data, test_data = self.preprocess(train_data, val_data, test_data)

        return [train_data, val_data, test_data]
    
class kadid10k_loader(database_loader):
    def __init__(self, download=True):
        super().__init__()
        self.url = 'https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip'
        self.dir = os.path.join(self.catalogue, 'kadid10k')
        self.images = os.path.join(self.dir, 'images')
        self.zip_file = os.path.join(self.catalogue, 'kadid10k.zip')
        self.distortion_mapping = {1: 'gblur', 2: 'lblur', 3: 'mblur', 4: 'cdiff', 5: 'cshift', # According to KADID-10k documentation
                                   6: 'cquant', 7: 'csat1', 8: 'csat2', 9: 'jp2k', 10: 'jpeg',
                                   11: 'wniose1', 12: 'wniose2', 13: 'inoise', 14: 'mnoise', 15: 'denoise',
                                   16: 'bright', 17: 'dark', 18: 'meanshft', 19: 'jit', 20: 'patch', 
                                   21: 'pixel', 22: 'quant', 23: 'cblock', 24: 'sharp', 25: 'contrst'} 
        if download:
            self.download(self.zip_file)
        elif not os.path.exists(self.zip_file):
            logging.info("Dataset not found. Downloading...")
            self.download(self.url, self.zip_file)
        
        if self.data_exist():
            logging.info("Patch files found. Loading patched data...")
            self.train = self.load_data('train')
            self.val = self.load_data('val')
            self.test= self.load_data('test')
        else:
            data = self.prepare_data()
            logging.info('Mapping data to TensorFlow format...')
            self.train = self.map2tf('training', data[0])
            self.val = self.map2tf('validation', data[1])
            self.test = self.map2tf('test', data[2])
            self.save_data(self.train, 'train')
            self.save_data(self.val, 'val')
            self.save_data(self.test, 'test')
        #self.train, self.val, self.test = self.encode(data)

        logging.info("Data loaded successfully.")

    def prepare_data(self, filter=True):
        data_path = os.path.join(self.dir, 'dmos.csv')
        data = pd.read_csv(data_path, header=0, usecols=[0, 2])
        data.columns = ['image', 'DMOS']
        data['distortion'] = data['image'].apply(lambda x: self.distortion_mapping.get(int(x.split('_')[1]), 'other'))
        #if filter:
            #data = data[data['distortion'].isin(self.distortion_mapping.values())]
        data.to_csv(os.path.join(self.dir,'dmos_with_names.csv'), index=False)

        train_data, val_data, test_data = self.split_data(data)
        train_data, val_data, test_data = self.preprocess(train_data, val_data, test_data)

        return [train_data, val_data, test_data]
        