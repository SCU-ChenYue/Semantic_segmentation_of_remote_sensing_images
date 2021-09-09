import cv2
from tqdm import tqdm

TEST_SET = ['1.tif']
# TEST_SET = ['1.png', '2.png', '3.png']


def print_seen(src, img, index):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 1:
                src[i, j, 0] = 0
                src[i, j, 1] = 0
                src[i, j, 2] = 200

            elif img[i, j, 0] == 2:
                src[i, j, 0] = 150
                src[i, j, 1] = 0
                src[i, j, 2] = 250

            elif img[i, j, 0] == 3:
                src[i, j, 0] = 150
                src[i, j, 1] = 150
                src[i, j, 2] = 200

            elif img[i, j, 0] == 4:
                src[i, j, 0] = 150
                src[i, j, 1] = 150
                src[i, j, 2] = 250

            elif img[i, j, 0] == 5:
                src[i, j, 0] = 0
                src[i, j, 1] = 200
                src[i, j, 2] = 0

            elif img[i, j, 0] == 6:
                src[i, j, 0] = 0
                src[i, j, 1] = 250
                src[i, j, 2] = 150

            elif img[i, j, 0] == 7:
                src[i, j, 0] = 150
                src[i, j, 1] = 200
                src[i, j, 2] = 150

            elif img[i, j, 0] == 8:
                src[i, j, 0] = 200
                src[i, j, 1] = 0
                src[i, j, 2] = 200

            elif img[i, j, 0] == 9:
                src[i, j, 0] = 250
                src[i, j, 1] = 0
                src[i, j, 2] = 150

            elif img[i, j, 0] == 10:
                src[i, j, 0] = 250
                src[i, j, 1] = 150
                src[i, j, 2] = 150

            elif img[i, j, 0] == 11:
                src[i, j, 0] = 0
                src[i, j, 1] = 200
                src[i, j, 2] = 250

            elif img[i, j, 0] == 12:
                src[i, j, 0] = 0
                src[i, j, 1] = 200
                src[i, j, 2] = 200

            elif img[i, j, 0] == 13:
                src[i, j, 0] = 200
                src[i, j, 1] = 0
                src[i, j, 2] = 0

            elif img[i, j, 0] == 14:
                src[i, j, 0] = 200
                src[i, j, 1] = 150
                src[i, j, 2] = 0

            elif img[i, j, 0] == 15:
                src[i, j, 0] = 250
                src[i, j, 1] = 200
                src[i, j, 2] = 0

    cv2.imwrite(('data/predict/out/%d.tif' % index), src)


for i in tqdm(range(len(TEST_SET))):
    path = TEST_SET[i]
    src = cv2.imread('data/test/' + path)
    img = cv2.imread('data/predict/pre/' + path)
    print_seen(src, img, i)


# ALL 0
# VEGETATION 1
# ROAD 4
# BUILDING 2
# WATER 3





