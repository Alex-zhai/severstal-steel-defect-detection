# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/9/19 21:29

from __future__ import division, absolute_import, print_function

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras.applications import Xception
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, UpSampling2D, Activation
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split


IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
PATIENCE = 7


def mask2rle(img):
    tmp = np.rot90(np.flipud(img), k=3)
    rle = []
    lastColor = 0
    startpos = 0
    endpos = 0
    tmp = tmp.reshape(-1, 1)
    for i in range(len(tmp)):
        if (lastColor == 0) and tmp[i] > 0:
            startpos = i
            lastColor = 1
        elif (lastColor == 1) and (tmp[i] == 0):
            endpos = i - 1
            lastColor = 0
            rle.append(str(startpos) + ' ' + str(endpos - startpos + 1))
    return " ".join(rle)


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height = imgshape[1]
    mask = np.zeros(width * height).astype(np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]
    return np.flipud(np.rot90(mask.reshape(height, width), k=1))


def rle2mask_eda(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def build_masks(rles, input_shape):
    depth = len(rles)
    masks = np.zeros((*input_shape, depth))

    for i, rle in enumerate(rles):
        if type(rle) is str:
            masks[:, :, i] = rle2mask(rle, input_shape)
    return masks


df_train = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
df_train = df_train[df_train['EncodedPixels'].notnull()].reset_index(drop=True)
train_image_ids = df_train['ImageId'].unique()

X_train, X_val = train_test_split(train_image_ids, test_size=0.2, random_state=42)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='../input/severstal-steel-defect-detection/train_images',
                 batch_size=32, dim=(256, 1600), n_channels=3,
                 n_classes=4, random_state=2019, shuffle=True, aug=False):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        self.seq = aug
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            return X, y.astype(int)

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            img_path = os.path.join(self.base_path, ID)
            img = self.__load_grayscale(img_path)
            # Store samples
            X[i,] = gray2rgb(img[:, :, 0])
        return X

    def __generate_y(self, list_IDs_batch):
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            rles = image_df['EncodedPixels'].values
            masks = build_masks(rles, input_shape=self.dim)

            y[i,] = masks
        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img







def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float64')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.8), 'float64')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# ref: https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10
        metric += iou
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.8], tf.float64)


def get_callback(patient):
    ES = EarlyStopping(
        monitor='loss',
        patience=patient,
        mode='max',
        verbose=1)
    RR = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=patient / 2,
        min_lr=0.000001,
        verbose=1,
        mode='max')
    return [ES, RR]


base_model = Xception(weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)
base_model.load_weights('../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

base_out = base_model.output
up1 = UpSampling2D(32, interpolation='bilinear')(base_out)
conv1 = Conv2D(1, (1, 1))(up1)
conv1 = Activation('sigmoid')(conv1)
model = Model(base_model.input, conv1)
print(model.summary())
model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[iou_metric])

history = model.fit_generator(keras_generator(BATCH_SIZE),
                              steps_per_epoch=len(df_train.index) / EPOCHS,
                              epochs=EPOCHS,
                              verbose=1,
                              shuffle=True,
                              callbacks=get_callback(PATIENCE)
                              )
# test
test_img = []
testfiles = os.listdir("../input/severstal-steel-defect-detection/test_images/")
for fn in testfiles:
    img = cv2.imread('../input/severstal-steel-defect-detection/test_images/' + fn)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    test_img.append(img)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(
    np.asarray(test_img),
    batch_size=BATCH_SIZE
)

testfiles = os.listdir("../input/severstal-steel-defect-detection/test_images/")
nb_samples = len(testfiles)
predict = model.predict_generator(test_generator, steps=nb_samples / BATCH_SIZE)

pred_rle = []
for img in predict:
    img = cv2.resize(img, (1600, 256))
    tmp = np.copy(img)
    tmp[tmp < 0.8] = 0
    tmp[tmp > 0] = 1
    pred_rle.append(mask2rle(tmp))

pred_rle_4 = []
for _ in pred_rle:
    pred_rle_4.extend([_, _, _, _])
len(pred_rle_4)

sub = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')
sub['EncodedPixels'] = pred_rle_4
sub.to_csv('submission.csv', index=False)
