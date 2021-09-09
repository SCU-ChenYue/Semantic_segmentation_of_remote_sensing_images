import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import cv2

"""
label information of 15 classes: 通道顺序和opencv读出来的相反
1 industrial land         200,    0,    0
2 urban residential       250,    0, 150
3 rural residential       200, 150, 150 *
4 traffic land            250, 150, 150
5 paddy field             0,     200,    0
6 irrigated land          150,  250,   0 *
7 dry cropland            150, 200, 150
8 garden plot             200,     0, 200
9 arbor woodland          150,     0, 250
10 shrub land              150,  150, 250
11 natural grassland       250,  200,    0
12 artificial grassland    200,  200,    0
13 river                   0,         0, 200
14 lake                    0,     150, 200
15 pond                    0,     200, 250
"""


filepath = 'data/newdataSource/'


def img_change(image, index):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, 2] == 200 and image[i, j, 1] == 0 and image[i, j, 0] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 1

            elif image[i, j, 2] == 250 and image[i, j, 1] == 0 and image[i, j, 0] == 150:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 2

            elif image[i, j, 2] == 200 and image[i, j, 1] == 150 and image[i, j, 0] == 150:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 3

            elif image[i, j, 2] == 250 and image[i, j, 1] == 150 and image[i, j, 0] == 150:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 4

            elif image[i, j, 2] == 0 and image[i, j, 1] == 200 and image[i, j, 0] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 5

            elif image[i, j, 2] == 150 and image[i, j, 1] == 250 and image[i, j, 0] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 6

            elif image[i, j, 2] == 150 and image[i, j, 1] == 200 and image[i, j, 0] == 150:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 7

            elif image[i, j, 2] == 200 and image[i, j, 1] == 0 and image[i, j, 0] == 200:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 8

            elif image[i, j, 2] == 150 and image[i, j, 1] == 0 and image[i, j, 0] == 250:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 9

            elif image[i, j, 2] == 150 and image[i, j, 1] == 150 and image[i, j, 0] == 250:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 10

            elif image[i, j, 2] == 250 and image[i, j, 1] == 200 and image[i, j, 0] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 11

            elif image[i, j, 2] == 200 and image[i, j, 1] == 200 and image[i, j, 0] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 12

            elif image[i, j, 2] == 0 and image[i, j, 1] == 0 and image[i, j, 0] == 200:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 13

            elif image[i, j, 2] == 0 and image[i, j, 1] == 150 and image[i, j, 0] == 200:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 14

            elif image[i, j, 2] == 0 and image[i, j, 1] == 200 and image[i, j, 0] == 250:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 15

            else:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 0

    cv2.imwrite('data/newdataSource/label_01/' + str(index) + '.tif', image)


# img = load_img(filepath + 'label/' + '1.tif')
# print(img)
for i in range(1, 11):
    img = cv2.imread(filepath + 'label/' + str(i) + '.tif')
    img_change(img, i)


