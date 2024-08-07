import numpy as np
import pandas as pd
import cv2
import urllib.request
import subprocess
import os
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


class tid2013_loader:
    def __init__(self, download=True, path='databases'):
        self.url = 'https://www.ponomarenko.info/tid2013/tid2013.rar'
        self.dir = os.path.join(path, 'tid2013')
        self.file_path = os.path.join(self.dir, 'tid2013.rar')
        self.winrar_path = 'D:\\Programy\\WinRAR\\WinRAR.exe'
        self.sevenzip_path = 'D:\\Programy\\7-Zip\\7z.exe'
        self.distortion_mapping = {1: 'wn', 2:'wnc', 3:'scn', 4:'mn', 5:'hfn', 6:'in', 7:'qn', 8: 'gblur', 9:'idn', 10: 'jpeg', 11: 'jp2k', 12:'jpegte', 13:'jp2kte'} # According to TID2013 documentation

        os.makedirs(self.dir, exist_ok=True)

        if download:
            self.download()
        if self.__patch_files_exist():
            print("Patch files found. Loading patched data...")
            data = self.__load_patched_data()
        else:
            data = self.prepare_data()
        self.train, self.val, self.test = self.encode(data)

    def __patch_files_exist(self):
        # Check if patch files are present in the directory
        return (os.path.exists(os.path.join(self.dir, 'normalized_distorted_images', 'training', 'patch_training.csv')) and
                os.path.exists(os.path.join(self.dir, 'normalized_distorted_images', 'validation', 'patch_validation.csv')) and
                os.path.exists(os.path.join(self.dir, 'normalized_distorted_images', 'test', 'patch_test.csv')))

    def __load_patched_data(self):
        # Load the existing patched data
        columns=['image', 'MOS', 'distortion']
        train_data = pd.read_csv(os.path.join(self.dir, 'normalized_distorted_images', 'training', 'patch_training.csv'))
        val_data = pd.read_csv(os.path.join(self.dir, 'normalized_distorted_images', 'validation', 'patch_validation.csv'))
        test_data = pd.read_csv(os.path.join(self.dir, 'normalized_distorted_images', 'test', 'patch_test.csv'))
        return [train_data, val_data, test_data]
    
    def download(self): 
        try:
            print(f"Downloading dataset from {self.url}...")
            urllib.request.urlretrieve(self.url, self.file_path)
            print(f"Dataset downloaded and saved as '{self.file_path}'")
        except Exception as e:
            print(f"Failed to download dataset: {e}.")
            return False  

        if not self.__extract_with_winrar():
            self.__extract_with_sevenzip()

    def __extract_with_winrar(self):
        try:
            print(f"Extracting dataset using WinRAR...")
            subprocess.run([self.winrar_path, 'x', self.file_path, self.dir], capture_output=True, text=True)
            print(f"Dataset extracted to '{self.dir}'.")
            return True
        except Exception as e:
            print(f"Error using WinRAR: {e}")
            return False

    def __extract_with_sevenzip(self):
        try:
            print(f"Extracting dataset using 7-Zip...")
            subprocess.run([self.sevenzip_path, 'x', '-aoa', self.file_path, f'-o{self.dir}'], capture_output=True, text=True)
            print(f"Dataset extracted to '{self.dir}'.")
            return True
        except Exception as e:
            print(f"Error using 7-Zip: {e}.")
            return False

    def split_data(self, data1):
        train_data, test_data = train_test_split(data1, test_size=0.2, random_state=40)
        train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=40)
        return train_data, val_data, test_data

    def __normalize_and_slice(self, train_data, val_data, test_data, patch_size=32, cross=False):
    
        def local_normalize(patch, P=3, Q=3, C=1):
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
                    patch_path = os.path.join(output_dir_patches, f"{os.path.splitext(image_filename)[0]}_patch_{patch_count}.bmp")
                    patch_filename = f"{os.path.splitext(image_filename)[0]}_patch_{patch_count}.bmp"
                    cv2.imwrite(patch_path, patch)
                    # Add patch info to the list
                    self.patch_info_list.append([patch_filename, mos_value, distortion])
                    patch_count += 1

        sets = [(train_data, 'training'), (val_data, 'validation'), (test_data, 'test')]
        print('Normalizing and slicing images...')

        for (data, name) in sets:
            output_dir_full = os.path.join(self.dir, 'normalized_distorted_images', name, 'full')
            output_dir_patches = os.path.join(self.dir, 'normalized_distorted_images', name, 'patches')
            norm_file_info_path = os.path.join(self.dir, 'normalized_distorted_images', name, f'norm_{name}.csv')
            patch_file_info_path = os.path.join(self.dir, 'normalized_distorted_images', name, f'patch_{name}.csv')
            os.makedirs(output_dir_full, exist_ok=True)
            os.makedirs(output_dir_patches, exist_ok=True)

            self.norm_info_list = []
            self.patch_info_list = []

            for row in data.itertuples(index=False):
                image_filename = row[0]
                mos_value = row[1]
                distortion = row[2]
                image_path = f'{self.dir}/distorted_images/{image_filename}'
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Failed to load image: {image_filename}")
                    continue

                # Normalize the image
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_normalized = local_normalize(image_gray)
                # Save
                image_filename = f'NORM_{image_filename}'
                self.norm_info_list.append([image_filename, mos_value, distortion])
                cv2.imwrite(os.path.join(output_dir_full, image_filename), image_normalized)
                # Slice to patches
                slice_image(image_normalized, patch_size)

            norm_info_df = pd.DataFrame(self.norm_info_list, columns=['image', 'MOS', 'distortion'])
            norm_info_df.to_csv(norm_file_info_path, index=False)
            print(f"[{name}]: Full image info listed in: {norm_file_info_path}.")
            patch_info_df = pd.DataFrame(self.patch_info_list, columns=['image', 'MOS', 'distortion'])
            patch_info_df.to_csv(patch_file_info_path, index=False)
            print(f"[{name}]: Patch info listed in: {patch_file_info_path}.")

    def prepare_data(self, filter=True):
        data_path = os.path.join(self.dir, 'mos_with_names.txt')
        data = pd.read_csv(data_path, header=None, delimiter=' ')
        data = data.iloc[:, [1, 0]]  # swap column order
        data.columns = ['image', 'MOS']
        # Specify distortion type based on filename
        data['distortion'] = data['image'].apply(lambda x: self.distortion_mapping.get(int(x.split('_')[1]), 'other'))
        if filter:
            data = data[data['distortion'].isin(self.distortion_mapping.values())]
        data.to_csv('databases/tid2013/mos_with_names.csv', index=False)

        train_data, val_data, test_data = self.split_data(data)
        self.__normalize_and_slice(train_data, val_data, test_data)

        train_data = pd.read_csv(f'{self.dir}/normalized_distorted_images/training/patch_training.csv')
        val_data = pd.read_csv(f'{self.dir}/normalized_distorted_images/validation/patch_validation.csv')
        test_data = pd.read_csv(f'{self.dir}/normalized_distorted_images/test/patch_test.csv')
        return [train_data, val_data, test_data]

    def encode(self, dataframes):
        for i in range(len(dataframes)):
            dists = dataframes[i]['distortion']
            le = LabelEncoder()
            y_class_encoded = le.fit_transform(dists)
            dists_one_hot = to_categorical(y_class_encoded, num_classes=13).astype(int)
            dataframes[i]['distortion_encoded'] = [np.array(one_hot) for one_hot in dists_one_hot]
            dataframes[i] = dataframes[i].drop(['distortion'], axis=1)
        return dataframes