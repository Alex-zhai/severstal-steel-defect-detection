# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/9/18 15:14

import tensorflow as tf
from keras.layers import Flatten
from keras.backend import pow


def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = Flatten()(y_true)
    y_pred_pos = Flatten()(y_pred)
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
