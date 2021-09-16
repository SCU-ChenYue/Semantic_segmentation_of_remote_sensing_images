import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import cv2
from tqdm import tqdm

"""
label information of 15 classes: 通道顺序和opencv读出来的相反
1 industrial land         200,    0,    0

"""


filepath = 'data/data_15classes/'


def img_change(image, index):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            if image[i, j, 2] == 200 and image[i, j, 1] == 150 and image[i, j, 0] == 150:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 1

            elif image[i, j, 2] == 150 and image[i, j, 1] == 250 and image[i, j, 0] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 2

            elif image[i, j, 2] == 150 and image[i, j, 1] == 0 and image[i, j, 0] == 250:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 3

            elif image[i, j, 2] == 250 and image[i, j, 1] == 200 and image[i, j, 0] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 4

            elif image[i, j, 2] == 0 and image[i, j, 1] == 0 and image[i, j, 0] == 200:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 5

            elif image[i, j, 2] == 0 and image[i, j, 1] == 150 and image[i, j, 0] == 200:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 6

            else:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 0

    cv2.imwrite('data/data_15classes/label_01/' + str(index) + '.png', image)


# img = load_img(filepath + 'label/' + '1.tif')
# print(img)
for i in tqdm(range(1, 11)):
    img = cv2.imread(filepath + 'label/' + str(i) + '.png')
    img_change(img, i)


