# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/9/23 14:54

from __future__ import division, absolute_import, print_function

import os
import random

import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.applications import Xception
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, UpSampling2D, Activation
from keras.losses import binary_crossentropy
from keras.models import Model
from sklearn.model_selection import KFold

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 20
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


df_train = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
df_train = df_train[df_train['EncodedPixels'].notnull()].reset_index(drop=True)


def imageId_2_img(image_index):
    fn = df_train['ImageId_ClassId'].iloc[image_index].split('_')[0]
    img = cv2.imread('../input/severstal-steel-defect-detection/train_images/' + fn)
    mask = rle2mask(df_train['EncodedPixels'].iloc[image_index], img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    return img, mask


# step2: get train, val data  k-fold
def get_train_and_val(train_index, valid_index):
    ids_train, ids_valid = df_train.index.values[train_index], df_train.index.values[valid_index]
    x_train, x_valid, y_train, y_valid = [], [], [], []

    for ids in ids_train:
        img, mask = imageId_2_img(ids)
        x_train.append(img)
        y_train.append(mask)

    for ids in ids_valid:
        img, mask = imageId_2_img(ids)
        x_valid.append(img)
        y_valid.append(mask)

    return np.array(x_train), np.array(x_valid), np.array(y_train), np.array(y_valid)


# step3: data augmentation
def flip_train_and_val(x_train, x_valid, y_train, y_valid):
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
    x_valid = np.append(x_valid, [np.fliplr(x) for x in x_valid], axis=0)
    y_valid = np.append(y_valid, [np.fliplr(x) for x in y_valid], axis=0)
    return x_train, x_valid, y_train, y_valid


def do_augmentation(X_train, y_train):
    # Use seq_det to build augmentation.
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.Noop(),
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.Noop(),
            iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=0),
            iaa.Noop(),
            iaa.PerspectiveTransform(scale=(0.04, 0.08)),
            iaa.Noop(),
            iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=0),
            iaa.Noop(),
            iaa.Flipud(),
            iaa.Noop(),
            iaa.ElasticTransformation(alpha=50, sigma=5)
        ])
    ])
    seq_det = seq.to_deterministic()
    X_train_aug = seq_det.augment_images(X_train)
    y_train_aug = seq_det.augment_images(y_train)
    return np.array(X_train_aug), np.array(y_train_aug)


def generator(features, labels, batch_size=BATCH_SIZE):
    # create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], features.shape[3]))
    batch_labels = np.zeros((batch_size, labels.shape[1], labels.shape[2], labels.shape[3]))

    while True:
        # Fill arrays of batch size with augmented data taken randomly from full passed arrays
        indexes = random.sample(range(len(features)), batch_size)
        print(indexes)
        # Perform the exactly the same augmentation for X and y
        random_augmented_images, random_augmented_labels = do_augmentation(features[indexes], labels[indexes])
        # random_augmented_images, random_augmented_labels = features[indexes], labels[indexes]
        batch_features[:, :, :, :] = random_augmented_images[:, :, :, :]
        batch_labels[:, :, :, :] = random_augmented_labels[:, :, :, :]
        yield batch_features, batch_labels


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


def get_test_imgs():
    test_img = []
    testfiles = os.listdir("../input/severstal-steel-defect-detection/test_images/")
    for fn in testfiles:
        img = cv2.imread('../input/severstal-steel-defect-detection/test_images/' + fn)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_img.append(img)
    return test_img


def train_and_eval_and_submit():
    base_model = Xception(weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)
    # base_model.load_weights('../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

    base_out = base_model.output
    up1 = UpSampling2D(32, interpolation='bilinear')(base_out)
    conv1 = Conv2D(1, (1, 1))(up1)
    conv1 = Activation('sigmoid')(conv1)
    model = Model(base_model.input, conv1)
    print(model.summary())
    model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[iou_metric])

    fold_count = 1
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1234)
    for train_index, valid_index in k_fold.split(df_train.index.values):
        x_train, x_valid, y_train, y_valid = get_train_and_val(train_index, valid_index)
        x_train, x_valid, y_train, y_valid = flip_train_and_val(x_train, x_valid, y_train, y_valid)
        y_train, y_valid = np.expand_dims(y_train, axis=-1), np.expand_dims(y_valid, axis=-1)
        print(x_train.shape, y_train.shape)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[x_valid, y_valid],
                  verbose=1, workers=10, use_multiprocessing=True, callbacks=get_callback(PATIENCE))
        # model.fit_generator(generator(x_train, y_train),
        #                    steps_per_epoch=200,
        #                    epochs=EPOCHS,
        #                    verbose=1,
        #                    shuffle=True,
        #                    validation_data=[x_valid, y_valid],
        #                    callbacks=get_callback(PATIENCE)
        #                    )
        # test
        if fold_count == 1:
            preds_test = predict_result(model, np.array(get_test_imgs()))
        else:
            preds_test += predict_result(model, np.array(get_test_imgs()))
        fold_count += 1

    avg_pred = [pred / 5 for pred in preds_test]
    pred_rle = []
    for img in avg_pred:
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


def predict_result(model, x_test):  # predict both orginal and reflect x
    preds_test1 = model.predict(x_test, batch_size=BATCH_SIZE).reshape(-1, IMG_SIZE, IMG_SIZE)
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    preds_test2_reflect = model.predict(x_test_reflect, batch_size=BATCH_SIZE).reshape(-1, IMG_SIZE, IMG_SIZE)
    preds_test2 = np.array([np.fliplr(x) for x in preds_test2_reflect])
    preds_avg = (preds_test1 + preds_test2) / 2
    return preds_avg


if __name__ == '__main__':
    train_and_eval_and_submit()
