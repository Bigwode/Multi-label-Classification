# coding:utf-8
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import backend as K
from keras import optimizers
import numpy as np
import argparse
import sys
import mulNet
import load_data

# dimensions of our images.
img_width, img_height = 224, 224

nb_train_samples = 4013
nb_test_samples = 4022
epochs = 5
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


base_model, model = mulNet.build_vgg_raw(img_width, img_height)


def train(X_train, X_test, y_train, y_test):

    train_datagen = ImageDataGenerator(
        rotation_range=30,
        # rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode="nearest"
    )

    train_generator = train_datagen.flow(X_train, y_train, batch_size=32, shuffle=True, seed=None)
    val_generator = ImageDataGenerator().flow(X_test, y_test, batch_size=32, shuffle=True, seed=None)

    print('训练顶层分类器')

    for layer in base_model.layers:
        layer.trainable = False

    opt = optimizers.Adam(lr=1e-3,decay=1e-6, amsgrad=True)
    model.compile(loss={'predict_class': 'categorical_crossentropy',
                        'predict_attri': 'binary_crossentropy'
                        },
                  optimizer=opt,  # 'rmsprop'
                  # loss_weights=[0.1, 0.9],
                  metrics=['accuracy']
                  )

    # history_t1 = model.fit(X_train, [y_train[:, :2], y_train[:, 2:]], epochs=epochs, batch_size=32,
    #           validation_data=(X_test, [y_test[:, :2], y_test[:, 2:]])
    #           # steps_per_epoch=nb_test_samples//batch_size,
    #           # validation_steps=nb_test_samples // batch_size * epochs
    #           )
    print('Starting to train...')
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in train_generator:
            cost = model.train_on_batch(x_batch, [y_batch[:, :2], y_batch[:, 2:]])
            batches += x_batch.shape[0]
            if batches > X_train.shape[0]:
                break
            print('batches:', batches, 'loss:', cost[0], 'loss_class:',cost[1], 'loss_attri:', cost[2], 'acc_class:', cost[3], 'acc_attr:', cost[4])
    print('Starting to test...')
    cost = model.evaluate(X_test, [y_test[:, :2], y_test[:, 2:]], batch_size=32)
    print('test loss: ', cost)
    # classes = model.predict_classes(X_test, batch_size=32)
    # proba = model.predict_proba(X_test, batch_size=32)
    model.save('first_blood.h5')
    # model.fit(X_train, [y_train[:, :2], y_train[:, 2:]], epochs=epochs, batch_size=32,
    #           validation_data=(X_test, [y_test[:, :2], y_test[:, 2:]])
    #           # steps_per_epoch=nb_test_samples//batch_size,
    #           # validation_steps=nb_test_samples // batch_size * epochs
    #
    #           )
'''
    print('对顶层分类器fine-tune')

    for layer in model.layers[:11]:
        layer.trainable = False
    for layer in model.layers[11:]:
        layer.trainable = True

    opt = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9)
    model.compile(loss={'predict_class':'categorical_crossentropy',
                        'predict_attri': 'binary_crossentropy'
                        },
                  optimizer=opt,  # 'rmsprop'
                  # loss_weights=[0.1, 0.9],
                  metrics=['accuracy'])

    # history_ft = model.fit_generator(
    #     train_generator,
    #     validation_data=val_generator,
    #     steps_per_epoch=nb_train_samples // batch_size,
    #     epochs=epochs)
    model.fit(X_train, [y_train[:, :2], y_train[:, 2:]], epochs=epochs, batch_size=32,
              validation_data=(X_test, [y_test[:, :2], y_test[:, 2:]])
              # steps_per_epoch=nb_test_samples // batch_size
              # validation_steps=nb_test_samples//batch_size,
              )

    # model.save('first_blood.h5')
    # plot_training(history_ft)
'''


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()



if __name__=='__main__':

    arg = argparse.ArgumentParser(description='Process the input_output path.')
    arg.add_argument("-train_data", "--train_data", default='./attributes_dataset/train/',
                     help="path to input train_dataset")
    arg.add_argument("-train_label", "--train_label", default='./attributes_dataset/train_label.txt',
                     help="path to input train_label")
    arg.add_argument("-test_data", "--test_data", default='./attributes_dataset/test/',
                     help="path to input test_dataset")
    arg.add_argument("-test_label", "--test_label", default='./attributes_dataset/test_label.txt',
                     help="path to input test_label")
    args = arg.parse_args()

    train_data_dir = vars(args)['train_data']
    train_label_dir = vars(args)['train_label']
    test_data_dir = vars(args)['test_data']
    test_label_dir = vars(args)['test_label']
    train_data, train_labels = load_data.load_data(img_width, img_height, train_data_dir, train_label_dir)
    test_data, test_labels = load_data.load_data(img_width, img_height, test_data_dir, test_label_dir)

    train(train_data, test_data, train_labels, test_labels)
    # score = model.evaluate(X_test, y_test, batch_size=32)
