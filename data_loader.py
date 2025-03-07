import numpy as np
import pandas as pd
import cv2
import urllib.request
import subprocess
import os
import logging
import configparser

from scipy.signal import convolve2d

### Set-up logs ###
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear() 

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)

### Set-up config file ###
def read_config(config_path, section='data_loader'):
    '''Reads parameters from the config file'''
    config = configparser.ConfigParser()
    config.read(config_path)
    if section not in config:
        raise ValueError(f"Section '{section}' not found in the config file.")
    
    elif section == 'data_loader':
        config_values = {
            'catalogue': config.get(section, 'catalogue'),
            'winrar': config.get(section, 'winrar'),
            '7zip': config.get(section, '7zip'),
        }
    
    return config_values
    
### Classes for managing, downloading and preprocessing databases ###
class database_loader:
    '''Parent class for database-specific loaders.'''
    def __init__(self):
        config_parameters = read_config('config.ini', section='data_loader') 
        self.catalogue = config_parameters['catalogue']
        self.winrar = config_parameters['winrar']
        self.sevenzip = config_parameters['7zip']

        ## Attributes to be declared within the child class ##
        self.url = ''           # URL of the dataset (if applicable)
        self.exdir = ''         # Directory where the exctracted dataset is stored
        self.measureName = ''   # MOS/DMOS column name
        self.images_dir = ''    # Directory where the images are stored
        self.archive_file = ''  # Path to the rar/zip file
        self.metadata = None    # Dataframe with filenames, mos/dmos and distortion types
        self.num_classes = None # Number of different distortions
        self.quality_clusters = None # Categories of quality (0 - very bad, 1 - bad, ...) 
        
    def data_exist(self):
        '''Check if patch files are present in the directory.'''
        return (os.path.exists(os.path.join(self.exdir, 'metadata.csv')) and os.path.exists(os.path.join(self.exdir, 'X.npy'))
        and os.path.exists(os.path.join(self.exdir, 'y_reg.npy')) and os.path.exists(os.path.join(self.exdir, 'y_class.npy')))
        
    
    def download(self, extract_in='databases'):
        '''Download the dataset from the URL and extract it to the directory.
        Args:
            extract_in (str, optional): Provide if the dataset is not extracted into a folder named after the file
        Note:
            You need to specify path into WinRAR or 7zip in  config.ini.
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
                logging.error(f"Error while exctracting: {e}")
                return False
            else:
                logging.info(f"Dataset extracted in '{self.exdir}'.")
                return True

        def extract_in_windows(extract_in='databases'):
            try:

                logging.info(f"Extracting {self.archive_file} with 7-Zip...")
                result = subprocess.run([self.sevenzip, 'x', '-aoa', self.archive_file, f'-o{extract_in}'], capture_output=True, text=True)
                if result.returncode != 0:
                    logging.info(f"Extracting {self.archive_file} with WinRAR...")
                    subprocess.run([self.winrar, 'x', self.archive_file, extract_in], capture_output=True, text=True)

            except Exception as e:
                logging.error(f"Error while exctracting: {e}")
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
            
    def cross(self, metadata, X, mapping, cross_mapping, as_training):
        filter_values = set(cross_mapping.values())
        distortion_dict = {key: value for key, value in mapping.items() if value in filter_values}

        # filter distortions
        indices = metadata[self.metadata['distortion'].isin(distortion_dict.keys())].index 
        metadata = metadata.loc[indices, :]
        X =  X[indices].copy()  
        metadata.reset_index(drop=True, inplace=True)

        if as_training:
            # map labels to test database
            value_to_label = {v: k for k, v in cross_mapping.items()} # test value: test label
            label_to_label = {k: value_to_label[v] for k, v in mapping.items() if v in value_to_label} # training label: test label
            metadata['distortion'] = metadata['distortion'].map(label_to_label)

        # sequantialize labels
        label_mapping = {label: index for index, label in enumerate(np.unique(metadata['distortion']))}
        metadata['distortion'] = metadata['distortion'].map(label_mapping)

        return metadata, X
    
    def preprocess(self, database, patch_size=32):
        total_images = len(database)
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
                    y_reg.append(score)
                    y_class.append(distortion)                    
                    patch_count += 1
        
        output_dir_patches = os.path.join(self.exdir, 'patches')
        os.makedirs(output_dir_patches, exist_ok=True)
        patches = []
        X = []
        y_reg = []
        y_class= []
        for idx, row in database.iterrows():
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
                filename = filename.lower()
                slice_image(image_normalized, patch_size)
                processed_images += 1  
                print(f'Preprocessed {processed_images}/{total_images} images.', end='\r') 
                
        patches = pd.DataFrame(patches, columns = ['image', self.measureName, 'distortion'])
        patches.to_csv(os.path.join(self.exdir, 'metadata.csv'), index=False)
        X = np.array(X)
        X = X[..., np.newaxis]
        y_reg = np.array(y_reg, dtype=np.float32)
        y_class = np.array(y_class, dtype=np.int64)
        np.save(os.path.join(self.exdir, 'X.npy'), X)
        np.save(os.path.join(self.exdir, 'y_reg.npy'), y_reg)
        np.save(os.path.join(self.exdir, 'y_class.npy'), y_class)
                          
        return patches, X, y_reg, y_class
                    
# TO DO:
# filter pristine images
# refine logger

class tid2013_loader(database_loader):
    def __init__(self):
        super().__init__()
        self.url = 'https://www.ponomarenko.info/tid2013/tid2013.rar'
        self.exdir = os.path.join(self.catalogue, 'tid2013')
        self.measureName = 'MOS'
        self.images_dir = os.path.join(self.exdir, 'distorted_images')
        self.archive_file = os.path.join(self.catalogue, 'tid2013.rar')
        self.distortion_mapping = {1: 'wn', 2:'wnc', 3:'scn', 4:'mn', 5:'hfn',  # According to TID2013 documentation
                                    6:'inoise', 7:'qn', 8: 'gblur', 9:'idn', 10: 'jpeg', 11: 'jp2k', 12:'jpegte', 13:'jp2kte',
                                    14:'non-eccnoise', 15:'loc_blk-wise_dists', 16:'meanshft', 17:'contrast', 18:'satur', 19:'mnoise',
                                    20:'cnoise', 21:'noise_compress', 22:'dith_quant', 23:'chrome', 24:'recon'}
                                     
        self.distortion_mapping_kadid10k = {11: 'wn', 12:'wnc', 13:'inoise', 1: 'gblur',
                                         15:'idn', 10: 'jpeg', 9: 'jp2k',
                                         18:'meanshft', 14:'mnoise', 6:'dith_quant'}

        self.quality_clusters = 7
            
        #self.distortion_mapping_live = {1: 'wn', 2:'wnc', 3:'scn', 4:'mn', 5:'hfn', 
        #                             6:'inoise', 7:'qn', 8: 'gblur', 9:'idn', 10: 'jpeg',
        #                             11: 'jp2k', 12:'jpegte', 13:'jp2kte'}
                                      
        self.num_classes = len(self.distortion_mapping)+1                                   
        os.makedirs(self.exdir, exist_ok=True)

        if not self.download(extract_in=self.exdir):
            logging.error("Failed to download or extract the database.")
            return

        if self.data_exist():
            logging.info("Loading data...")
            self.metadata = pd.read_csv(os.path.join(self.exdir, 'metadata.csv'))
            self.X = np.load(os.path.join(self.exdir, 'X.npy'))
            self.y_reg = np.load(os.path.join(self.exdir, 'y_reg.npy'))
            self.y_class = np.load(os.path.join(self.exdir, 'y_class.npy'))                
        else:
            self.metadata, self.X, self.y_reg, self.y_class = self.prepare_data()
        logging.info("TID2013 loaded successfully.")                                       
        
    def prepare_data(self):
        data_path = os.path.join(self.exdir, 'mos_with_names.txt')
        database = pd.read_csv(data_path, header=None, delimiter=' ')
        database = database.iloc[:, [1, 0]]  # swap column order
        database.columns = ['image', 'MOS']
        database['distortion'] = [int(img.split('_')[1]) for img in database['image']]
        database = self.preprocess(database)
        return database
    
class kadid10k_loader(database_loader): 
    def __init__(self, filter, as_training=False):
        super().__init__()
        self.url = 'https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip'
        self.exdir = os.path.join(self.catalogue, 'kadid10k')
        self.measureName = 'DMOS'
        self.images_dir = os.path.join(self.exdir, 'images')
        self.archive_file = os.path.join(self.catalogue, 'kadid10k.zip')
        self.distortion_mapping = {1: 'gblur', 2: 'lblur', 3: 'mblur', 4: 'cdiff', 5: 'cshift', # According to KADID-10k documentation
                                6: 'dith_quant', 7: 'csat1', 8: 'csat2', 9: 'jp2k', 10: 'jpeg',
                                11: 'wn', 12: 'wnc', 13: 'inoise', 14: 'mnoise', 15: 'idn',
                                16: 'bright', 17: 'dark', 18: 'meanshft', 19: 'jit', 20: 'patch', 
                                21: 'pixel', 22: 'quant', 23: 'cblock', 24: 'sharp', 25: 'contrst'}
                                
        self.distortion_mapping_tid2013 = {8: 'gblur', 22: 'dith_quant',  11: 'jp2k', 10: 'jpeg', 1: 'wn', 
                                        2: 'wnc', 9: 'idn', 6: 'inoise', 19:'mnoise', 16:'meanshft'} 
                                                                          
        self.num_classes = len(self.distortion_mapping)+1                                                                     
        self.quality_clusters = 5 
        if not self.download():
            logging.error("Failed to download or extract the database.")
            return

        if self.data_exist():
            logging.info("Loading data...")
            self.metadata = pd.read_csv(os.path.join(self.exdir, 'metadata.csv'))
            self.X = np.load(os.path.join(self.exdir, 'X.npy'))
            self.y_reg = np.load(os.path.join(self.exdir, 'y_reg.npy'))
            self.y_class = np.load(os.path.join(self.exdir, 'y_class.npy')) 

            #if (filter != None) and (filter != 'kadid10k'):             
            #    self.metadata, self.X = self.cross(self.metadata, self.X, self.distortion_mapping, self.distortion_mapping_tid2013, as_training=as_training)
            #    self.y_reg = np.array(self.metadata[self.measureName], dtype=np.float32)  
            #    self.y_class = np.array(self.metadata['distortion'], dtype=np.int64)
            #    self.num_classes = len(self.distortion_mapping_tid2013)+1     

        else:
            self.metadata, self.X, self.y_reg, self.y_class = self.prepare_data()
        logging.info("KADID-10k loaded successfully.")        

    def prepare_data(self):
        database_path = os.path.join(self.exdir, 'dmos.csv')
        database = pd.read_csv(database_path, header=0, usecols=[0, 2])
        database.columns = ['image', 'DMOS']
        database['distortion'] = [int(img.split('_')[1]) for img in database['image']]
        database = self.preprocess(database)
        return database