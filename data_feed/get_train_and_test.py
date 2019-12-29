# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/9/18 14:46

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from utils import utils
from sklearn.model_selection import train_test_split

img_w = 800  # resized weidth
img_h = 256  # resized height
batch_size = 12
epochs = 25
# batch size for training unet


# load full data and label no mask as -1
train_df = pd.read_csv(os.path.join("/export/sdb/home/zhaijianwei/severstal-steel-defect-detection/data/", 'train.csv')).fillna(-1)
# image id and class id are two seperate entities and it makes it easier to split them up in two columns
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
# lets create a dict with class id and encoded pixels and group all the defaults per image
train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis=1)
grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)

# create a dict of all the masks
masks = {}
for index, row in train_df[train_df['EncodedPixels'] != -1].iterrows():
    masks[row['ImageId_ClassId']] = row['EncodedPixels']

train_image_ids = train_df['ImageId'].unique()

X_train, X_val = train_test_split(train_image_ids, test_size=0.2, random_state=42)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_ids, labels, image_dir, batch_size=32,
                 img_h=256, img_w=512, shuffle=True):
        self.list_ids = list_ids
        self.labels = labels
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids)) / self.batch_size)

    def __getitem__(self, index):
        'generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # get list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]
        # generate data
        X, y = self.__data_generation(list_ids_temp)
        # return data
        return X, y

    def on_epoch_end(self):
        'update ended after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        'generate data containing batch_size samples'
        X = np.empty((self.batch_size, self.img_h, self.img_w, 1))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 4))

        for idx, id in enumerate(list_ids_temp):
            file_path = os.path.join(self.image_dir, id)
            image = cv2.imread(file_path, 0)
            image_resized = cv2.resize(image, (self.img_w, self.img_h))
            image_resized = np.array(image_resized, dtype=np.float64)
            # standardization of the image
            image_resized -= image_resized.mean()
            image_resized /= image_resized.std()

            mask = np.empty((img_h, img_w, 4))

            for idm, image_class in enumerate(['1', '2', '3', '4']):
                rle = self.labels.get(id + '_' + image_class)
                # if there is no mask create empty mask
                if rle is None:
                    class_mask = np.zeros((1600, 256))
                else:
                    class_mask = utils.rle_to_mask(rle, width=1600, height=256)

                class_mask_resized = cv2.resize(class_mask, (self.img_w, self.img_h))
                mask[..., idm] = class_mask_resized
            X[idx,] = np.expand_dims(image_resized, axis=2)
            y[idx,] = mask

        # normalize Y
        y = (y > 0).astype(int)
        return X, y


def get_train_and_val_generator():
    params = {'img_h': img_h,
              'img_w': img_w,
              'image_dir': "/export/sdb/home/zhaijianwei/severstal-steel-defect-detection/data/train_images",
              'batch_size': batch_size,
              'shuffle': True}

    # Get Generators
    training_generator = DataGenerator(X_train, masks, **params)
    validation_generator = DataGenerator(X_val, masks, **params)
    return training_generator, validation_generator