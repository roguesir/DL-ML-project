# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
import os
import numpy as np


def load_dataset(filedir):
    """
    读取数据
    :param filedir:
    :return:
    """
    image_data_list = []
    label = []
    train_image_list = os.listdir(filedir)
    for img in train_image_list:
        url = os.path.join(filedir + os.sep + img)
        image = load_img(url, grayscale=True, target_size=(48, 48))
        #print(image.shape)
        image_data_list.append(img_to_array(image))
        label.append(img.split('.')[0].split("-")[1])
    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data, label


def make_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))    
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.save('./data/model/model-8.h5')
    return model


if __name__ == '__main__':
    train_loss = []
    train_accuracy = []
    train_x, train_y = load_dataset('data/train_data')
    test_x, test_y = load_dataset('data/test_data')
    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y)    
    model = make_network()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    hist_train = model.fit(train_x, train_y, batch_size=500, epochs=100, verbose=1)
    print(hist_train.history)
    model.evaluate(test_x, test_y, batch_size = 100)
    #print(hist_test.history)




