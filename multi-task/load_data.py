# coding:utf-8
import keras
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import os
import cv2

CLASS_NUM = 9
names = ['is_male', 'have long_hair', 'have glasses', 'have hat', 'have T-shirt',
         'have long_sleeves', 'have shorts', 'have jeans', 'have long_pants']


def load_label(label_dir):
    labels = []
    with open(label_dir) as f:
        for line in f:
            label = []
            # number = int(line.split(' ')[0].split('.')[0])
            # print(number)

            attr = line.strip('\n').split(' ')[5:]
            for attribute in attr:
                if attribute != '':  # train_labels 229行 coordinate:NaN
                    if int(attribute) != 1:
                        attribute = 0
                    label.append(int(attribute))
            labels.append(np.array(label))
    # print(labels)
    return labels


def load_data(img_width, img_height, data_dir, label_dir):
    data = []
    labels = []
    print("[INFO] loading labels...")
    imagePaths = sorted(list(paths.list_images(data_dir)))
    names = load_label(label_dir)
    # print(names)

    # 这样写就不能打乱顺序
    paths_labels = list(zip(imagePaths,names))
    random.seed(42)
    random.shuffle(paths_labels)
    imagePaths, names = zip(*paths_labels)
    print("[INFO] loading images...")

    for imagePath in imagePaths:
        p = imagePaths.index(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (img_width, img_height))
        image = img_to_array(image)
        data.append(image)

        # name = int(imagePath.split('/')[-1].split('.')[0])
        label = names[int(p)]
        labels.append(label)


    # scale to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    one_hot_labels = keras.utils.to_categorical(labels[:, 0], num_classes=2)
    labels = np.hstack((one_hot_labels, labels[:, 1:]))
    print(labels.shape)
    return data, labels


# data_dir = './attributes_dataset/train/'
# label_dir = './attributes_dataset/train_label.txt'
# load_data(224, 224, data_dir, label_dir)

