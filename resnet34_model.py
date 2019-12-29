# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/9/19 17:39

from distutils.version import StrictVersion

import cv2
import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.engine.topology import get_source_inputs
from keras.layers import Input
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.utils import get_file
from keras.preprocessing.image import ImageDataGenerator

if StrictVersion(keras.__version__) < StrictVersion('2.2.0'):
    from keras.applications.imagenet_utils import _obtain_input_shape
else:
    from keras_applications.imagenet_utils import _obtain_input_shape

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


df_train = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
df_train = df_train[df_train['EncodedPixels'].notnull()].reset_index(drop=True)


def keras_generator(batch_size):
    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            fn = df_train['ImageId_ClassId'].iloc[i].split('_')[0]
            img = cv2.imread('../input/severstal-steel-defect-detection/train_images/' + fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = rle2mask(df_train['EncodedPixels'].iloc[i], img.shape)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)


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


# https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/unet/models.py

def handle_block_names_old(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name


def Upsample2D_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2),
                     batchnorm=False, skip=None):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names_old(stage)

        x = layers.UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate()([x, skip])

        x = layers.Conv2D(filters, kernel_size, padding='same', name=conv_name + '1')(x)
        if batchnorm:
            x = layers.BatchNormalization(name=bn_name + '1')(x)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        x = layers.Conv2D(filters, kernel_size, padding='same', name=conv_name + '2')(x)
        if batchnorm:
            x = layers.BatchNormalization(name=bn_name + '2')(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        return x

    return layer


def Transpose2D_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2),
                      transpose_kernel_size=(4, 4), batchnorm=False, skip=None):
    def layer(input_tensor):

        conv_name, bn_name, relu_name, up_name = handle_block_names_old(stage)

        x = layers.Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,
                                   padding='same', name=up_name)(input_tensor)
        if batchnorm:
            x = layers.BatchNormalization(name=bn_name + '1')(x)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        if skip is not None:
            x = layers.Concatenate()([x, skip])

        x = layers.Conv2D(filters, kernel_size, padding='same', name=conv_name + '2')(x)
        if batchnorm:
            x = layers.BatchNormalization(name=bn_name + '2')(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)

        return x

    return layer


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def basic_identity_block(filters, stage, block):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        x = layers.Add()([x, input_tensor])
        return x

    return layer


def basic_conv_block(filters, stage, block, strides=(2, 2)):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)
        shortcut = x
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        shortcut = layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = layers.Add()([x, shortcut])
        return x

    return layer


def usual_identity_block(filters, stage, block):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)
        x = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '3')(x)
        x = layers.Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)(x)

        x = layers.Add()([x, input_tensor])
        return x

    return layer


def usual_conv_block(filters, stage, block, strides=(2, 2)):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)
        shortcut = x
        x = layers.Conv2D(filters, (1, 1), name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '2', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '3')(x)
        x = layers.Conv2D(filters * 4, (1, 1), name=conv_name + '3', **conv_params)(x)

        shortcut = layers.Conv2D(filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = layers.Add()([x, shortcut])
        return x

    return layer


def build_resnet(
        repetitions=(2, 2, 2, 2),
        include_top=True,
        input_tensor=None,
        input_shape=None,
        classes=1000,
        block_type='usual'):
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=101,
                                      data_format='channels_last',
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    if block_type == 'basic':
        conv_block = basic_conv_block
        identity_block = basic_identity_block
    else:
        conv_block = usual_conv_block
        identity_block = usual_identity_block

    # resnet bottom
    x = layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnet body
    for stage, rep in enumerate(repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = conv_block(filters, stage, block, strides=(1, 1))(x)

            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2))(x)

            else:
                x = identity_block(filters, stage, block)(x)

    x = layers.BatchNormalization(name='bn1', **bn_params)(x)
    x = layers.Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = layers.GlobalAveragePooling2D(name='pool1')(x)
        x = layers.Dense(classes, name='fc1')(x)
        x = layers.Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x)

    return model


weights_collection = [
    # ResNet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000.h5',
        'name': 'resnet34_imagenet_1000.h5',
        'md5': '2ac8277412f65e5d047f255bcbd10383',
    },

    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        'name': 'resnet34_imagenet_1000_no_top.h5',
        'md5': '8caaa0ad39d927cb8ba5385bf945d582',
    },
]


def find_weights(weights_collection, model_name, dataset, include_top):
    w = list(filter(lambda x: x['model'] == model_name, weights_collection))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    return w


def load_model_weights(weights_collection, model, dataset, classes, include_top):
    weights = find_weights(weights_collection, model.name, dataset, include_top)

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = get_file(weights['name'],
                                weights['url'],
                                cache_subdir='models',
                                md5_hash=weights['md5'])

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))


def ResNet34(input_shape, input_tensor=None, weights=None, classes=1000, include_top=True):
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         repetitions=(3, 4, 6, 3),
                         classes=classes,
                         include_top=include_top,
                         block_type='basic')
    model.name = 'resnet34'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model


def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = BatchActivate(x)
    return x


def squeeze_excite_block(input, filters):
    init = input
    se_shape = (1, 1, filters)
    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = layers.multiply([init, se])
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    se = squeeze_excite_block(x, num_filters)
    x = layers.add([se, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


# Build model
def build_model(input_shape=(256, 256, 3), DropoutRatio=0.5):
    input_layer = layers.Input(shape=input_shape)

    base_model = ResNet34(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=input_layer)

    conv4 = base_model.get_layer("stage4_unit1_relu1").output
    conv3 = base_model.get_layer("stage3_unit1_relu1").output
    conv2 = base_model.get_layer("stage2_unit1_relu1").output
    conv1 = base_model.get_layer("relu0").output

    mid = base_model.get_layer("relu1").output

    # 4 -> 8
    deconv4 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(mid)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(DropoutRatio)(uconv4)
    uconv4 = layers.Conv2D(256, (3, 3), padding="same")(uconv4)
    uconv4 = residual_block(uconv4, 256)
    uconv4 = residual_block(uconv4, 256, True)

    # 8 -> 16
    deconv3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(DropoutRatio)(uconv3)
    uconv3 = layers.Conv2D(128, (3, 3), padding="same")(uconv3)
    uconv3 = residual_block(uconv3, 128)
    uconv3 = residual_block(uconv3, 128, True)

    # 16 -> 32
    deconv2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = layers.Conv2D(64, (3, 3), padding="same")(uconv2)
    uconv2 = residual_block(uconv2, 64)
    uconv2 = residual_block(uconv2, 64, True)

    # 32 -> 64
    deconv1 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = layers.Conv2D(32, (3, 3), padding="same")(uconv1)
    uconv1 = residual_block(uconv1, 32)
    uconv1 = residual_block(uconv1, 32, True)

    # 64 -> 128
    output_layer = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding="same")(uconv1)
    output_layer = layers.Conv2D(1, (1, 1), strides=(1, 1), padding="same")(output_layer)
    output_layer = layers.Activation('sigmoid')(output_layer)

    model = Model(input_layer, output_layer)
    return model


model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3))
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

