# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import cv2
import random
import os
from tqdm import tqdm

filepath = '../data/newdataSource/'
test_data = []
test_label = []
n_label = 4 + 1
classes = [0., 1., 2., 3., 4.]

labelencoder = LabelEncoder()
labelencoder.fit(classes)


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        # img = np.array(img, dtype="float") / 255.0
    return img


# batch_size = 3
# for i in range(3):
#     label = load_img(filepath + 'label/' + str(i) + '.png', grayscale=True)
#     label = img_to_array(label).reshape((256 * 256,))  # 65536 x 1
#     test_label.append(label)
#     img = load_img(filepath + 'src/' + str(i) + '.png')
#     img = img_to_array(img)
#     test_data.append(img)

# test_label = np.array(test_label)
# print(test_label.shape)  # 65536 x 3
# test_label = np.array(test_label).flatten()
# print(test_label.shape)  # 196608 x 1
# print(test_label[111000:111100])
# test_label = labelencoder.transform(test_label)
# print(test_label[111000:111100])
# test_label = to_categorical(test_label, num_classes=n_label)
# print(test_label.shape)
# test_label = test_label.reshape((batch_size, 256 * 256, n_label))
# print(test_label.shape)
#
# test_data = np.array(test_data)
# print(test_data.shape)
label = cv2.imread(filepath + 'label_01/' + '1.tif', cv2.IMREAD_GRAYSCALE)
print(label.shape)
print(label)
