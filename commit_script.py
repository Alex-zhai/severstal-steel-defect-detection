# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/9/18 17:24

from __future__ import division, absolute_import, print_function

from keras.backend import pow
from keras import optimizers
from skimage import morphology
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Add, BatchNormalization, Activation
from keras.models import Model
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import keras
import cv2

from keras import backend as K
from sklearn.model_selection import train_test_split

img_w = 800  # resized weidth
img_h = 256  # resized height
batch_size = 12
epochs = 1
k_size = 3
load_pretrained_model = False
pretrained_model_path = ""

# load full data and label no mask as -1
train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
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


def rle_to_mask(rle_string, height, width):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters:
    rleString (str): Description of arg1
    height (int): height of the mask
    width (int): width of the mask

    Returns:
    numpy.array: numpy array of the mask
    '''
    rows, cols = height, width
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        for index, length in rlePairs:
            index -= 1
            img[index:index + length] = 255
        img = img.reshape(cols, rows)
        img = img.T
        return img


# Thanks to the authors of: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask_to_rle(mask):
    '''
    Convert a mask into RLE

    Parameters:
    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background

    Returns:
    sring: run length encoding
    '''
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class DataGenerator(keras.utils.Sequence):
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
                    class_mask = rle_to_mask(rle, width=1600, height=256)

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
              'image_dir': "../input/severstal-steel-defect-detection/train_images",
              'batch_size': batch_size,
              'shuffle': True}
    # Get Generators
    training_generator = DataGenerator(X_train, masks, **params)
    validation_generator = DataGenerator(X_val, masks, **params)
    return training_generator, validation_generator


def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return pow((1 - pt_1), gamma)


def bn_act(x, act=True):
    'batch normalization layer with an optinal activation layer'
    x = BatchNormalization()(x)
    if act:
        x = Activation('relu')(x)
    return x


def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    'convolutional layer which always uses the batch normalization layer'
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output


def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    res = conv_block(x, filters, k_size, padding, strides)
    res = conv_block(res, filters, k_size, padding, 1)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c


def ResUNet(img_h, img_w):
    f = [16, 32, 64, 128, 256]
    inputs = Input((img_h, img_w, 1))

    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = Conv2D(4, (1, 1), padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs)
    return model


train_generator, validation_generator = get_train_and_val_generator()

model = ResUNet(img_h=img_h, img_w=img_w)
adam = optimizers.Adam(lr=0.05, epsilon=0.1)
model.compile(optimizer=adam, loss=focal_tversky_loss, metrics=[tversky])
if load_pretrained_model:
    try:
        model.load_weights(pretrained_model_path)
        print('pre-trained model loaded!')
    except OSError:
        print('You need to run the model and load the trained model')

history = model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=epochs, verbose=1)
model.save("./ResUNetSteel_w800*256_base.h5")

submission = []

# get all files using glob
test_files = [f for f in
              glob.glob('../input/severstal-steel-defect-detection/test_images/' + "*.jpg",
                        recursive=True)]


# return tensor in the right shape for prediction
def get_test_tensor(img_dir, img_h, img_w, channels=1):
    X = np.empty((1, img_h, img_w, channels))
    # Store sample
    image = cv2.imread(img_dir, 0)
    image_resized = cv2.resize(image, (img_w, img_h))
    image_resized = np.array(image_resized, dtype=np.float64)
    # normalize image
    image_resized -= image_resized.mean()
    image_resized /= image_resized.std()

    X[0,] = np.expand_dims(image_resized, axis=2)

    return X


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


# a function to apply all the processing steps necessery to each of the individual masks
def process_pred_mask(pred_mask):
    pred_mask = cv2.resize(pred_mask.astype('float32'), (1600, 256))
    pred_mask = (pred_mask > .5).astype(int)
    pred_mask = remove_small_regions(pred_mask, 0.02 * np.prod(512)) * 255
    pred_mask = mask_to_rle(pred_mask)

    return pred_mask


# loop over all the test images
for f in test_files:
    # get test tensor, output is in shape: (1, 256, 512, 3)
    test = get_test_tensor(f, img_h, img_w)
    # get prediction, output is in shape: (1, 256, 512, 4)
    pred_masks = model.predict(test)
    # get a list of masks with shape: 256, 512
    pred_masks = [pred_masks[0][..., i] for i in range(0, 4)]
    # apply all the processing steps to each of the mask
    pred_masks = [process_pred_mask(pred_mask) for pred_mask in pred_masks]
    # get our image id
    id = f.split('/')[-1]
    # create ImageId_ClassId and get the EncodedPixels for the class ID, and append to our submissions list
    [submission.append((id + '_%s' % (k + 1), pred_mask)) for k, pred_mask in enumerate(pred_masks)]

# convert to a csv
submission_df = pd.DataFrame(submission, columns=['ImageId_ClassId', 'EncodedPixels'])
# check out some predictions and see if RLE looks ok
submission_df[submission_df['EncodedPixels'] != ''].head()

submission_df.to_csv('./base_submission.csv', index=False)
