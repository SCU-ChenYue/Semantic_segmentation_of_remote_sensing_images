import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    Dropout, ActivityRegularization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers.merge import concatenate
from utils import WarmUpCosineDecayScheduler
from Unet import unet_plus_plus, unet_new, unet_new2
import cv2
import random
import os
from keras.optimizers import Adam
import keras

matplotlib.use("Agg")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7
np.random.seed(seed)

# data_shape = 360*480
img_w = 256
img_h = 256
# 有一个为背景
n_label = 15 + 1
classes = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
labelencoder = LabelEncoder()
labelencoder.fit(classes)
image_sets = ['1.png', '2.png', '3.png']


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img, dtype="float") / 255.0
    return img


filepath = '../data/unet_buildings/unet_train/'


def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'src'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label).flatten()
                train_label = labelencoder.transform(train_label)
                train_label = to_categorical(train_label, num_classes=n_label)
                train_label = train_label.reshape((batch_size, img_w * img_h, n_label))
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0
            # data for validation


def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'src/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))
            # print label.shape
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label).flatten()
                valid_label = labelencoder.transform(valid_label)
                valid_label = to_categorical(valid_label, num_classes=n_label)
                valid_label = valid_label.reshape((batch_size, img_w * img_h, n_label))
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0


# 记得修改keras源文件中输出格式 channel first
def unet():
    inputs = Input((3, img_w, img_h))
    conv1 = Conv2D(32, (3, 3), padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128, (3, 3), padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, (3, 3), padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(256, (3, 3), padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(256, (3, 3), padding="same")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, (3, 3), padding="same")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, (3, 3), padding="same")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, (3, 3), padding="same")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(32, (3, 3), padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    # conv10 = Conv2D(n_label, (1, 1), activation="sigmoid")(conv9)
    # conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)
    conv10 = Conv2D(n_label, (1, 1), strides=(1, 1), padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv11 = Reshape((n_label, img_w * img_h))(conv10)
    conv12 = Permute((2, 1))(conv11)
    conv12 = Activation('softmax')(conv12)
    model = Model(inputs=inputs, outputs=conv12)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


learning_rate = 0.001
num_epochs = 50


def scheduler(epoch):
    if epoch < num_epochs * 0.4:
        return learning_rate
    elif epoch < num_epochs * 0.8:
        return learning_rate * 0.3
    else:
        return learning_rate * 0.03


def train(args):
    EPOCHS = num_epochs
    BS = 17
    train_set, val_set = get_train_val()
    train_numb = len(train_set)
    valid_numb = len(val_set)

    total_steps = EPOCHS * train_numb / BS
    warmup_steps = int(total_steps * 0.2)
    # model = unet_new(num_classes=n_label, input_shape=(3, img_w, img_h), lr_init=0.002, lr_decay=0.0000025)
    # model = unet()
    model = keras.models.load_model('../checkpoint/unet09.h5')
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model = unet_new(n_label, (3, 256, 256), 0.0015, 0.000007)
    # model = unet_new2(pretrained_weights='../checkpoint/unet06.h5', classNum=n_label, learning_rate=0.002, decay=0.000002)

    # model = keras.models.load_model('../checkpoint/unet04.h5')
    # model.compile()
    change_Lr = LearningRateScheduler(scheduler)
    modelcheck = ModelCheckpoint(args['model'], monitor='val_accuracy', save_best_only=True, mode='max')
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto',
    #                               epsilon=0.0001, cooldown=0, min_lr=0)

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=1e-3,
                                            total_steps=total_steps,
                                            warmup_learning_rate=4e-06,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=5)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=6, verbose=1, mode='max')
    callable = [modelcheck, change_Lr]

    print("the number of train data is", train_numb)
    print("the number of val data is", valid_numb)
    H = model.fit_generator(generator=generateData(BS, train_set), steps_per_epoch=train_numb // BS, epochs=EPOCHS,
                            verbose=1,
                            validation_data=generateValidData(BS, val_set), validation_steps=valid_numb // BS,
                            callbacks=callable, max_q_size=1)

    # plot the training loss and accuracy

    model.save('../checkpoint/unet03.h5')
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    print(H.history.keys())
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", help="training data's path",
                    default=True)
    ap.add_argument("-m", "--model", required=True,
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot0.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = args_parse()
    filepath = args['data']
    train(args)
    # predict()
