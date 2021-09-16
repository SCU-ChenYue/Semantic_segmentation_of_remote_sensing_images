import cv2
from tqdm import tqdm

"""
各地表标签的颜色通道值
1 building: 0 128 128
2 forest: 0 128 0
3 grass: 0 0 128
4 water: 128 0 0
5 wetland: 128 0 128
6 lake： 128 128 0
"""

filepath = 'data/after_process_label/'


def img_change(image, index):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, 0] == 0 and image[i, j, 1] == 128 and image[i, j, 2] == 128:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 1

            elif image[i, j, 0] == 0 and image[i, j, 1] == 128 and image[i, j, 2] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 2

            elif image[i, j, 0] == 0 and image[i, j, 1] == 0 and image[i, j, 2] == 128:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 3

            elif image[i, j, 0] == 128 and image[i, j, 1] == 0 and image[i, j, 2] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 4

            elif image[i, j, 0] == 128 and image[i, j, 1] == 0 and image[i, j, 2] == 128:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 5

            elif image[i, j, 0] == 128 and image[i, j, 1] == 128 and image[i, j, 2] == 0:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 6

            else:
                image[i, j, 0] = image[i, j, 1] = image[i, j, 2] = 0

    cv2.imwrite('data/after_process_label/label_01/' + str(index) + '.png', image)


for i in tqdm(range(1, 169)):
    img = cv2.imread(filepath + 'newLabel/' + str(i) + '.png')
    img_change(img, i)


