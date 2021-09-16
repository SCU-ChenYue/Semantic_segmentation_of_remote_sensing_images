import cv2
import random
import numpy as np
import os
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# TEST_SET = ['1.tif']
image_size = 256
classes = [0., 1., 2., 3., 4., 5., 6.]
labelencoder = LabelEncoder()
labelencoder.fit(classes)
all_image_path = 'data/after_process_label/ImageTest/'
out_path = 'data/after_process_label/predictResultLabel/'


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
                    help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())
    return args


def predict_all_files(file_dir, outPath):
    file_name_list = []
    for root, dirs, files in os.walk(file_dir):
        file_name_list = files
    file_name_list.sort(key=lambda x: int(x[:-4]))
    print(file_name_list)
    predict(file_name_list, outPath)


def predict(file_list, outPath):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model('checkpoint/unet03.h5')
    stride = image_size
    for n in range(len(file_list)):
        path = file_list[n]
        print(path)
        # load the image
        image = cv2.imread(all_image_path + path)
        h, w, _ = image.shape
        padding_h = (h // stride + 1) * stride
        padding_w = (w // stride + 1) * stride
        padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        print('src:', padding_img.shape)
        mask_whole = np.zeros((padding_h, padding_w), dtype=np.uint8)
        for i in range(padding_h // stride):
            for j in range(padding_w // stride):
                crop = padding_img[:3, i * stride:i * stride + image_size, j * stride:j * stride + image_size]
                _, ch, cw = crop.shape
                if ch != 256 or cw != 256:
                    print('invalid size!')
                    continue

                crop = np.expand_dims(crop, axis=0)
                # print 'crop:',crop.shape
                # pred = model.predict_classes(crop, verbose=2)

                pred = model.predict(crop)
                pred = np.argmax(pred, axis=-1)
                pred = labelencoder.inverse_transform(pred[0])
                # print (np.unique(pred))
                pred = pred.reshape((256, 256)).astype(np.uint8)
                # print 'pred:',pred.shape
                mask_whole[i * stride:i * stride + image_size, j * stride:j * stride + image_size] = pred[:, :]

        cv2.imwrite(outPath + str(n) + '.png', mask_whole[0:h, 0:w])


if __name__ == '__main__':
    # args = args_parse()
    predict_all_files(all_image_path, out_path)
