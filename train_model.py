# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/9/18 15:18

from data_feed.get_train_and_test import get_train_and_val_generator
from model.res_unet import ResUNet
from loss.focal_tversky_loss import tversky, focal_tversky_loss
from utils.utils import mask_to_rle
from keras import optimizers
from skimage import morphology
import cv2
import glob
import pandas as pd
import numpy as np




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


img_w = 800  # resized weidth
img_h = 256  # resized height
batch_size = 12
epochs = 25
load_pretrained_model = False
pretrained_model_path = ""

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
model.save("outputs/ResUNetSteel_w800*256_base.h5")

submission = []

# get all files using glob
test_files = [f for f in
              glob.glob('/export/sdb/home/zhaijianwei/severstal-steel-defect-detection/data/test_images/' + "*.jpg", recursive=True)]


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
